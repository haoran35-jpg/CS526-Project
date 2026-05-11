#!/usr/bin/env python3
"""Generate N-layer stacked HLO-style egglog, run egglog --to-json, primal graph,
treewidth upper bounds; append rows to `experiments/scale_results.csv`."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import networkx as nx
from networkx.algorithms.approximation import (
    treewidth_min_degree,
    treewidth_min_fill_in,
)


ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "experiments" / "generated"
RESULTS_CSV = ROOT / "experiments" / "scale_results.csv"
GEN_DIR.mkdir(parents=True, exist_ok=True)


PRELUDE = r"""
;; Auto-generated: stacked HLO transformer layers.

(datatype Hlo
  (HParam String)
  (HConst String)
  (HAdd Hlo Hlo)
  (HSub Hlo Hlo)
  (HMul Hlo Hlo)
  (HDiv Hlo Hlo)
  (HMax Hlo Hlo)
  (HExp Hlo)
  (HNeg Hlo)
  (HRelu Hlo)
  (HDot Hlo Hlo)
  (HTranspose Hlo)
  (HReshape Hlo String)
  (HBroadcast Hlo String)
  (HReduceSum Hlo String)
  (HReduceMax Hlo String)
  (FusedSoftmax Hlo String)
  (FusedGemmBias Hlo Hlo Hlo)
  (FusedGemmBiasRelu Hlo Hlo Hlo)
  (FusedAttn Hlo Hlo Hlo String)
  (FusedAttnResidual Hlo Hlo Hlo Hlo String)
  (FusedFFNResidual Hlo Hlo Hlo Hlo Hlo))

(rewrite (HTranspose (HTranspose x)) x)
(rewrite (HReshape (HReshape x s1) s2) (HReshape x s2))
(rewrite (HRelu (HRelu x)) (HRelu x))
(rewrite (HDot (HDot a b) c) (HDot a (HDot b c)))
(rewrite (HDot a (HDot b c)) (HDot (HDot a b) c))

(rewrite (HAdd (HDot a b) bias) (FusedGemmBias a b bias))
(rewrite (FusedGemmBias a b bias) (HAdd (HDot a b) bias))
(rewrite (HRelu (FusedGemmBias a b bias)) (FusedGemmBiasRelu a b bias))
(rewrite (FusedGemmBiasRelu a b bias) (HRelu (FusedGemmBias a b bias)))

(rewrite
  (HDiv
    (HExp (HSub x (HBroadcast (HReduceMax x axis) baxis)))
    (HBroadcast
      (HReduceSum
        (HExp (HSub x (HBroadcast (HReduceMax x axis) baxis)))
        axis)
      baxis))
  (FusedSoftmax x axis))
(rewrite
  (FusedSoftmax x axis)
  (HDiv
    (HExp (HSub x (HBroadcast (HReduceMax x axis) "axis_bcast")))
    (HBroadcast
      (HReduceSum
        (HExp (HSub x (HBroadcast (HReduceMax x axis) "axis_bcast")))
        axis)
      "axis_bcast")))

(rewrite (HDot (FusedSoftmax (HDot q (HTranspose k)) axis) v)
         (FusedAttn q k v axis))
(rewrite (FusedAttn q k v axis)
         (HDot (FusedSoftmax (HDot q (HTranspose k)) axis) v))
(rewrite (HAdd x (FusedAttn q k v axis)) (FusedAttnResidual x q k v axis))
(rewrite (FusedAttnResidual x q k v axis) (HAdd x (FusedAttn q k v axis)))

(rewrite (HAdd y (FusedGemmBias w2 (FusedGemmBiasRelu w1 y b1) b2))
         (FusedFFNResidual y w1 b1 w2 b2))
(rewrite (FusedFFNResidual y w1 b1 w2 b2)
         (HAdd y (FusedGemmBias w2 (FusedGemmBiasRelu w1 y b1) b2)))
"""


def layer_text(i: int, prev: str) -> str:
    return f"""
;; ---- layer {i} ----
(let $Q{i}    (HParam "Q{i}"))
(let $K{i}    (HParam "K{i}"))
(let $V{i}    (HParam "V{i}"))
(let $W1_{i}  (HParam "W1_{i}"))  (let $b1_{i} (HParam "b1_{i}"))
(let $W2_{i}  (HParam "W2_{i}"))  (let $b2_{i} (HParam "b2_{i}"))

(let $scores_{i}    (HDot $Q{i} (HTranspose $K{i})))
(let $smax_max_{i}  (HReduceMax $scores_{i} "axis_last"))
(let $smax_bmax_{i} (HBroadcast $smax_max_{i} "axis_bcast"))
(let $smax_shft_{i} (HSub $scores_{i} $smax_bmax_{i}))
(let $smax_exp_{i}  (HExp $smax_shft_{i}))
(let $smax_sum_{i}  (HReduceSum $smax_exp_{i} "axis_last"))
(let $smax_bsum_{i} (HBroadcast $smax_sum_{i} "axis_bcast"))
(let $smax_{i}      (HDiv $smax_exp_{i} $smax_bsum_{i}))
(let $attn_{i}      (HDot $smax_{i} $V{i}))
(let $y_{i}         (HAdd ${prev} $attn_{i}))

(let $h1_{i}  (HRelu (HAdd (HDot $W1_{i} $y_{i})  $b1_{i})))
(let $ffn_{i} (HAdd (HDot $W2_{i} $h1_{i}) $b2_{i}))
(let $out_{i} (HAdd $y_{i} $ffn_{i}))
"""


def gen_program(n_layers: int, run_iters: int) -> str:
    chunks = [PRELUDE, '\n(let $X (HParam "X"))\n']
    prev = "X"
    for i in range(1, n_layers + 1):
        chunks.append(layer_text(i, prev))
        prev = f"out_{i}"
    chunks.append(f"\n(run {run_iters})\n")
    return "".join(chunks)


def build_egglog() -> Path:
    subprocess.run(
        ["cargo", "build", "--release", "--quiet"], cwd=ROOT, check=True
    )
    out = subprocess.check_output(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=ROOT,
    )
    target = json.loads(out)["target_directory"]
    return Path(target) / "release" / "egglog"


def build_primal(egraph: dict) -> nx.Graph:
    nodes = egraph.get("nodes", {})
    g = nx.Graph()
    for n in nodes.values():
        parent = n.get("eclass")
        if parent is None:
            continue
        kid_classes = []
        for cid in n.get("children", []):
            child = nodes.get(cid)
            if child is not None and "eclass" in child:
                kid_classes.append(child["eclass"])
        verts = [parent] + kid_classes
        g.add_nodes_from(verts)
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if verts[i] != verts[j]:
                    g.add_edge(verts[i], verts[j])
    return g


def measure_treewidth(g: nx.Graph, max_classes_for_fill_in: int) -> dict:
    n = g.number_of_nodes()
    if n == 0:
        return dict(tw_min_degree=0, tw_min_fill_in=0, tw_upper_bound=0,
                    tw_md_secs=0.0, tw_mf_secs=0.0)

    t0 = time.perf_counter()
    tw1, _ = treewidth_min_degree(g)
    t1 = time.perf_counter()

    if n <= max_classes_for_fill_in:
        tw2, _ = treewidth_min_fill_in(g)
        t2 = time.perf_counter()
    else:
        tw2 = -1
        t2 = t1

    upper = tw1 if tw2 < 0 else min(tw1, tw2)
    return dict(
        tw_min_degree=tw1,
        tw_min_fill_in=tw2,
        tw_upper_bound=upper,
        tw_md_secs=round(t1 - t0, 3),
        tw_mf_secs=round(t2 - t1, 3),
    )


def run_one(egglog: Path, n_layers: int, run_iters: int,
            max_classes_for_fill_in: int) -> dict:
    src = gen_program(n_layers, run_iters)
    egg_path = GEN_DIR / f"hlo_layers_{n_layers:04d}.egg"
    egg_path.write_text(src)

    t0 = time.perf_counter()
    subprocess.run(
        [
            str(egglog),
            "--to-json",
            "--max-functions", "1000",
            "--max-calls-per-function", "10000000",
            str(egg_path),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    t_egg = round(time.perf_counter() - t0, 3)

    json_path = egg_path.with_suffix(".json")
    with open(json_path) as f:
        eg = json.load(f)

    t0 = time.perf_counter()
    g = build_primal(eg)
    t_build = round(time.perf_counter() - t0, 3)

    tw_stats = measure_treewidth(g, max_classes_for_fill_in)

    return {
        "n_layers": n_layers,
        "run_iters": run_iters,
        "e_nodes": len(eg.get("nodes", {})),
        "e_classes": g.number_of_nodes(),
        "primal_edges": g.number_of_edges(),
        **tw_stats,
        "egglog_secs": t_egg,
        "primal_secs": t_build,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--layers", type=int, nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    ap.add_argument("--run-iters", type=int, default=3)
    ap.add_argument(
        "--max-classes-for-fill-in", type=int, default=4000,
        help="skip min-fill-in heuristic above this many e-classes",
    )
    args = ap.parse_args()

    egglog = build_egglog()
    if not egglog.exists():
        print(f"egglog binary missing at {egglog}", file=sys.stderr)
        return 1

    rows = []
    for n in args.layers:
        print(f"==> N={n} layers")
        try:
            row = run_one(egglog, n, args.run_iters,
                          args.max_classes_for_fill_in)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode().strip().splitlines()
            print(f"    [error] {err[-1] if err else e}")
            continue
        rows.append(row)
        print(
            "    e_nodes={e_nodes:>7}  classes={e_classes:>7}  "
            "edges={primal_edges:>8}  tw<= {tw_upper_bound:>3}  "
            "(egg {egglog_secs}s  primal {primal_secs}s  "
            "md {tw_md_secs}s  mf {tw_mf_secs}s)".format(**row)
        )

    if rows:
        new_file = not RESULTS_CSV.exists()
        with open(RESULTS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new_file:
                w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nresults appended to {RESULTS_CSV}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

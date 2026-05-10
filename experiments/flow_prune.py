#!/usr/bin/env python3
"""Network-flow-guided e-graph saturation for the TASO-Sensat kernel.

Implements Algorithm 1 from
"ILP Constraint Analysis via Network Flow for E-Graph Saturation":

    repeat:                                                           # outer loop
        --- Saturation phase ---
        (saturate (run algebraic))                                    # via egglog
        (run tiling)                                                  # via egglog

        --- Network-flow analysis phase ---
        for each root e-class r:
            selection  = greedy_dag_extract(r)        # one e-node per used e-class
            schedule   = dfs_post_order(selection)    # program-point order
            live_per_p = liveness_analysis(schedule)
            Φ(L_p)     = Σ_{v in L_p} onchip(v, T_v)
            violations = {p : Φ(L_p) > S_max}

        for each violation (p, L_p, v* = argmax onchip in L_p):
            append   (subsume <synth_term(v*)>)   to the .egg

    until no violations remain (or `--rounds` reached)

The egglog .egg is regenerated each round with the accumulated
`(subsume ...)` actions inserted *before* the `(run-schedule ...)`
trailer, so the next saturation respects every prior pruning.

Usage:
    python3 experiments/flow_prune.py
        [--kernel experiments/kernels/TASO-Sensat.egg]
        [--smax 65536] [--sz 4]
        [--rounds 10]
        [--workdir experiments/_flow_prune]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.setrecursionlimit(50000)

# --------------------------------------------------------------------------- #
# Defaults pinned to the design we agreed on.
# --------------------------------------------------------------------------- #
DEFAULT_KERNEL = "experiments/kernels/TASO-Sensat.egg"
DEFAULT_SMAX   = 64 * 1024     # 65536 bytes on-chip budget
DEFAULT_SZ     = 4             # fp32
TILE_SIZES     = (32, 64, 128) # synthetic-mode tile menu
PII_TILE_MAX   = 64            # ACT PII tile menu top end (rows)
INF            = float("inf")

# Vocabulary-agnostic spec for tiled ops:
#   op_name : (untiled_op_name, child_idx_of_tile_size)
TILE_OP_SPEC = {
    "TiledMm":    ("Mm",    2),     # TASO-Sensat datatype
    "TiledConv":  ("Conv",  2),
    "TiledHDot":  ("HDot",  3),     # HLO datatype: 4th child is tile size
    "TiledHConv": ("HConv", 3),
    "PTiledGemm":    ("PGemm",    2),  # ACT PII datatype
    "PTiledDot":     ("PDot",     2),
    "PTiledAdd":     ("PAdd",     2),
    "PTiledSoftmax": ("PSoftmax", 2),
    "TiledDot":      ("Dot",      2),  # ACT IR2IR (egglog rendition, act_ir2ir.egg)
    "TiledAdd":      ("Add",      2),
}
TILED_OPS   = set(TILE_OP_SPEC.keys())
UNTILED_OPS = {u for (u, _) in TILE_OP_SPEC.values()}

# Greedy extractor cost model:
#   - heavily penalise the *untiled* Mm/Conv/HDot/HConv so the extractor
#     commits to a tile choice;
#   - bias the initial round toward the LARGEST tile (most aggressive
#     compute / most expensive on-chip), so that violations actually fire
#     and the prune loop has work to do.
COST_UNTILED_LA   = 100_000
COST_NON_TILED_OP = 1
def cost_for_tile(T: int) -> int:
    return 1 + (max(TILE_SIZES) - T)         # T=128 ⇒ 1, T=64 ⇒ 65, T=32 ⇒ 97


def tile_size_of(n: dict, nodes: dict) -> int | None:
    """Return tile size T for a tiled e-node, or None if not tiled."""
    spec = TILE_OP_SPEC.get(n["op"])
    if spec is None:
        return None
    _, idx = spec
    return int(nodes[n["children"][idx]]["op"])


# --------------------------------------------------------------------------- #
# egglog binary discovery (mirrors experiments/run.sh logic).
# --------------------------------------------------------------------------- #
def find_egglog_bin(repo_root: Path) -> Path:
    env_bin = os.environ.get("EGGLOG_BIN")
    if env_bin and Path(env_bin).is_file():
        return Path(env_bin)
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=repo_root, check=True, capture_output=True, text=True,
    ).stdout
    target_dir = Path(json.loads(out)["target_directory"])
    candidate = target_dir / "release" / "egglog"
    if not candidate.is_file():
        subprocess.run(
            ["cargo", "build", "--release", "--quiet"],
            cwd=repo_root, check=True,
        )
    if not candidate.is_file():
        sys.exit(f"egglog binary not found at {candidate}")
    return candidate


# --------------------------------------------------------------------------- #
# Kernel I/O.
# --------------------------------------------------------------------------- #
def render_kernel(base_egg: str, extras: list[str]) -> str:
    """Insert accumulated `(subsume ...)` actions just before the *first*
    real `(run-schedule ...)` (skipping comments)."""
    lines = base_egg.splitlines(keepends=True)
    insert_at = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(";"):
            continue
        if stripped.startswith("(run-schedule"):
            insert_at = i
            break

    block = ("\n".join([";; ----- flow_prune extras (round subsumes) -----", *extras]) + "\n\n"
             if extras else "")
    if insert_at is None:
        return base_egg + ("\n\n" + block if extras else "")
    return "".join(lines[:insert_at]) + block + "".join(lines[insert_at:])


def run_egglog(egglog_bin: Path, egg_path: Path) -> Path:
    json_path = egg_path.with_suffix(".json")
    res = subprocess.run(
        [str(egglog_bin), "--to-json",
         "--max-functions", "10000",
         "--max-calls-per-function", "1000000",
         str(egg_path)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        sys.stderr.write(res.stdout + "\n" + res.stderr + "\n")
        raise RuntimeError(f"egglog failed on {egg_path}")
    if not json_path.exists():
        raise RuntimeError(f"egglog did not produce {json_path}")
    return json_path


# --------------------------------------------------------------------------- #
# E-graph helpers.
# --------------------------------------------------------------------------- #
def is_primitive(nid: str) -> bool:
    return nid.startswith("primitive-")


def index_eclasses(nodes: dict) -> dict[str, list[str]]:
    eclass_to_nodes: dict[str, list[str]] = defaultdict(list)
    for nid, n in nodes.items():
        if n.get("subsumed", False):
            continue
        eclass_to_nodes[n["eclass"]].append(nid)
    return eclass_to_nodes


def find_root_eclasses(eg: dict) -> list[str]:
    """Roots = e-classes that never appear as a child of any e-node.

    These correspond to the program's outputs (the let-bound values that
    nobody consumes inside the kernel).
    """
    nodes = eg["nodes"]
    all_eclasses = set()
    child_eclasses = set()
    for n in nodes.values():
        if n.get("subsumed", False):
            continue
        all_eclasses.add(n["eclass"])
        for child_nid in n["children"]:
            child = nodes.get(child_nid)
            if child is not None:
                child_eclasses.add(child["eclass"])
    return sorted(all_eclasses - child_eclasses)


def node_self_cost(nid: str, n: dict, nodes: dict) -> float:
    op = n["op"]
    if op in UNTILED_OPS:
        return COST_UNTILED_LA
    T = tile_size_of(n, nodes)
    if T is not None:
        return float(cost_for_tile(T))
    return float(COST_NON_TILED_OP)


# --------------------------------------------------------------------------- #
# Greedy DAG extract.
# --------------------------------------------------------------------------- #
def greedy_extract(eg: dict, root_eclass: str) -> tuple[dict[str, str], dict[str, float]]:
    """Pick one e-node per used e-class via tree-cost DP (cycle-safe).

    Returns:
      eclass_choice : eclass_id -> chosen node_id (DAG selection)
      eclass_cost   : eclass_id -> tree cost of that selection
    """
    nodes = eg["nodes"]
    eclass_to_nodes = index_eclasses(nodes)

    eclass_cost: dict[str, float] = {}
    eclass_choice: dict[str, str] = {}

    def visit(c: str, stack: set[str]) -> float:
        if c in eclass_cost:
            return eclass_cost[c]
        if c in stack:
            return INF                                     # cycle ⇒ unusable here
        stack.add(c)
        best = (INF, None)
        for nid in eclass_to_nodes.get(c, []):
            n = nodes[nid]
            cost = node_self_cost(nid, n, nodes)
            ok = True
            for child_nid in n["children"]:
                child_class = nodes[child_nid]["eclass"]
                ck = visit(child_class, stack)
                if ck >= INF:
                    ok = False
                    break
                cost += ck
            if ok and cost < best[0]:
                best = (cost, nid)
        stack.discard(c)
        eclass_cost[c] = best[0]
        eclass_choice[c] = best[1]
        return best[0]

    visit(root_eclass, set())
    return eclass_choice, eclass_cost


# --------------------------------------------------------------------------- #
# DAG realisation + schedule + liveness.
# --------------------------------------------------------------------------- #
def realize_dag(root_eclass: str, eclass_choice: dict[str, str], nodes: dict) -> tuple[str, dict[str, list[str]]]:
    root_nid = eclass_choice[root_eclass]
    if root_nid is None:
        raise RuntimeError(f"no extractable selection for root e-class {root_eclass}")
    children_of: dict[str, list[str]] = {}
    seen: set[str] = set()
    stack = [root_nid]
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        kids = []
        for child_nid in nodes[nid]["children"]:
            child_class = nodes[child_nid]["eclass"]
            chosen = eclass_choice.get(child_class, child_nid)
            if chosen is None:
                continue
            kids.append(chosen)
            stack.append(chosen)
        children_of[nid] = kids
    return root_nid, children_of


def dfs_post_order(root_nid: str, children_of: dict[str, list[str]]) -> list[str]:
    order: list[str] = []
    visited: set[str] = set()

    def go(nid: str) -> None:
        if nid in visited:
            return
        visited.add(nid)
        for c in children_of.get(nid, []):
            go(c)
        order.append(nid)

    go(root_nid)
    return order


def liveness_analysis(schedule: list[str], children_of: dict[str, list[str]], root_nid: str) -> list[set[str]]:
    """Return live set per program point (interval = [def, last_use])."""
    pos = {nid: i for i, nid in enumerate(schedule)}
    consumers: dict[str, list[str]] = defaultdict(list)
    for parent, kids in children_of.items():
        for k in kids:
            consumers[k].append(parent)

    last_use: dict[str, int] = {}
    last_p = len(schedule) - 1
    for nid in schedule:
        cs = consumers.get(nid, [])
        if not cs or nid == root_nid:
            last_use[nid] = last_p                  # outputs live to the end
        else:
            last_use[nid] = max(pos[p] for p in cs if p in pos)

    live_per_p: list[set[str]] = []
    live: set[str] = set()
    for p, nid in enumerate(schedule):
        live.add(nid)
        live = {v for v in live if last_use[v] >= p}
        live_per_p.append(set(live))
    return live_per_p


# --------------------------------------------------------------------------- #
# Footprint and violation detection.
# --------------------------------------------------------------------------- #
def _strip_quotes(s: str) -> str:
    return s[1:-1] if len(s) >= 2 and s[0] == '"' and s[-1] == '"' else s


def onchip_of(nid: str, nodes: dict, sz: int,
              meta: dict | None = None,
              eclass_choice: dict[str, str] | None = None) -> int:
    """On-chip footprint of e-node `nid`.

    Two modes:
      * meta is None  → legacy synthetic 3·T²·sz model (TASO-Sensat / HLO).
      * meta provided → ACT PII model: only `PNamed` wrappers carry footprint;
        for tiled inner ops, scale by (T / max(TILE_SIZES)).
    """
    n = nodes[nid]
    if meta is None:
        T = tile_size_of(n, nodes)
        return 0 if T is None else 3 * T * T * sz

    if n["op"] != "PNamed":
        return 0
    name = _strip_quotes(nodes[n["children"][0]]["op"])
    info = meta.get(name)
    if not info or info.get("onchip_bytes", 0) == 0:
        return 0
    base = info["onchip_bytes"]
    inner_class = nodes[n["children"][1]]["eclass"]
    inner_nid = (eclass_choice or {}).get(inner_class) or n["children"][1]
    inner = nodes[inner_nid]
    T = tile_size_of(inner, nodes)
    if T is None:
        return base
    return max(1, int(base * T / PII_TILE_MAX))


def _actionable_inner(v: str, nodes: dict, eclass_choice: dict[str, str] | None) -> str | None:
    """If `v` is a PNamed wrapper whose chosen inner op has tile alternatives,
    return that inner e-node id (an actionable subsume target). Else None."""
    n = nodes[v]
    if n["op"] != "PNamed":
        return v if (n["op"] in TILED_OPS or n["op"] in UNTILED_OPS) else None
    inner_class = nodes[n["children"][1]]["eclass"]
    inner_nid = (eclass_choice or {}).get(inner_class) or n["children"][1]
    inner = nodes[inner_nid]
    if inner["op"] in TILED_OPS or inner["op"] in UNTILED_OPS:
        return inner_nid
    return None


def find_violations(schedule, live_per_p, nodes, smax, sz,
                    meta: dict | None = None,
                    eclass_choice: dict[str, str] | None = None):
    """One violation record per offending program point.

    `v_star` is restricted to e-nodes that have *tile alternatives* (so
    subsuming them lets the next saturation pick a smaller-tile variant).
    Live tensors with no tile alternative (loads/moves/etc.) contribute to Φ
    but are never themselves subsumed."""
    out = []
    for p, live in enumerate(live_per_p):
        contrib = [(v, onchip_of(v, nodes, sz, meta, eclass_choice)) for v in live]
        phi = sum(c for _, c in contrib)
        if phi <= smax:
            continue
        actionable = []
        for v, c in contrib:
            if c <= 0:
                continue
            inner = _actionable_inner(v, nodes, eclass_choice)
            if inner is not None:
                actionable.append((inner, c))
        v_star = max(actionable, key=lambda x: x[1])[0] if actionable else None
        out.append({
            "p": p,
            "phi": phi,
            "v_star": v_star,            # may be None (no actionable target)
            "live": set(live),
        })
    return out


# --------------------------------------------------------------------------- #
# subsume command synthesis (with `let` sharing to keep .egg size bounded).
# --------------------------------------------------------------------------- #
def _let_name(nid: str) -> str:
    return "$_cls_" + nid.replace("-", "_").replace(".", "_")


def emit_subsume_with_lets(
    v_star: str,
    nodes: dict,
    eclass_choice: dict[str, str],
    seen_let_nids: set[str],
    extras: list[str],
) -> str:
    """Append any newly-needed `(let $_cls_… expr)` bindings, then append a
    `(subsume (Op …))` that references those names.

    Returns the subsume command (also already in `extras`)."""

    def child_arg(child_nid: str) -> str:
        if is_primitive(child_nid):
            return nodes[child_nid]["op"]            # literal already in surface syntax
        child_class = nodes[child_nid]["eclass"]
        chosen = eclass_choice.get(child_class, child_nid) or child_nid
        emit_lets(chosen)
        return _let_name(chosen)

    def emit_lets(nid: str) -> None:
        if nid in seen_let_nids or is_primitive(nid):
            return
        n = nodes[nid]
        op = n["op"]
        for child_nid in n["children"]:                # children first (post-order)
            if not is_primitive(child_nid):
                child_class = nodes[child_nid]["eclass"]
                chosen = eclass_choice.get(child_class, child_nid) or child_nid
                emit_lets(chosen)
        seen_let_nids.add(nid)
        if not n["children"]:
            expr = f"({op})"
        else:
            expr = "(" + op + " " + " ".join(child_arg(c) for c in n["children"]) + ")"
        extras.append(f"(let {_let_name(nid)} {expr})")

    n = nodes[v_star]
    op = n["op"]
    args = " ".join(child_arg(c) for c in n["children"])
    cmd = f"(subsume ({op} {args}))" if n["children"] else f"(subsume ({op}))"
    extras.append(cmd)
    return cmd


# --------------------------------------------------------------------------- #
# Outer loop.
# --------------------------------------------------------------------------- #
def network_flow_guided_saturation(
    egglog_bin: Path,
    base_egg_path: Path,
    workdir: Path,
    smax: int,
    sz: int,
    max_outer: int,
    max_subsumes_per_round: int = 8,
    meta: dict | None = None,
):
    base_egg = base_egg_path.read_text()
    extras: list[str] = []
    seen_subsumes: set[str] = set()
    seen_let_nids: set[str] = set()
    history: list[dict] = []
    quiescent_rounds = 0

    workdir.mkdir(parents=True, exist_ok=True)

    for outer in range(max_outer):
        cur_egg = workdir / f"round{outer:02d}.egg"
        cur_egg.write_text(render_kernel(base_egg, extras))
        json_path = run_egglog(egglog_bin, cur_egg)
        eg = json.loads(json_path.read_text())
        nodes = eg["nodes"]

        roots = find_root_eclasses(eg)
        all_violations: list[tuple[dict, dict[str, str]]] = []
        per_root: list[dict] = []
        for r in roots:
            choice, costs = greedy_extract(eg, r)
            if choice.get(r) is None:
                continue
            root_nid, children_of = realize_dag(r, choice, nodes)
            schedule = dfs_post_order(root_nid, children_of)
            live_per_p = liveness_analysis(schedule, children_of, root_nid)
            vs = find_violations(schedule, live_per_p, nodes, smax, sz,
                                 meta=meta, eclass_choice=choice)
            per_root.append({
                "root": r,
                "schedule_len": len(schedule),
                "tree_cost": costs.get(r, INF),
                "max_phi": max((v["phi"] for v in vs), default=0),
                "n_violations": len(vs),
            })
            for v in vs:
                all_violations.append((v, choice))

        new_subsumes = 0
        n_unactionable = sum(1 for v, _ in all_violations if v["v_star"] is None)
        if all_violations:
            offenders: dict[str, tuple[int, str, dict[str, str]]] = {}
            for v, choice in all_violations:
                if v["v_star"] is None:
                    continue
                key = v["v_star"]
                cur = offenders.get(key)
                if cur is None or v["phi"] > cur[0]:
                    offenders[key] = (v["phi"], v["v_star"], choice)
            ranked = sorted(offenders.values(), key=lambda t: -t[0])[:max_subsumes_per_round]
            for phi, v_star, choice in ranked:
                cmd = emit_subsume_with_lets(v_star, nodes, choice, seen_let_nids, extras)
                if cmd not in seen_subsumes:
                    seen_subsumes.add(cmd)
                    new_subsumes += 1

        history.append({
            "round": outer,
            "n_roots": len(roots),
            "n_violations": sum(p["n_violations"] for p in per_root),
            "n_unactionable": n_unactionable,
            "max_phi": max((p["max_phi"] for p in per_root), default=0),
            "new_subsumes": new_subsumes,
            "total_subsumes": len(seen_subsumes),
            "n_enodes": len(nodes),
            "n_eclasses": len({n["eclass"] for n in nodes.values()}),
        })

        print(
            f"round {outer:>2} | roots={len(roots):>2} | "
            f"violations={sum(p['n_violations'] for p in per_root):>3} "
            f"(unactionable={n_unactionable}) | "
            f"max Φ={max((p['max_phi'] for p in per_root), default=0):>7} B | "
            f"+subsume={new_subsumes} (total {len(seen_subsumes)}) | "
            f"|V|={len(nodes)} |classes|={len({n['eclass'] for n in nodes.values()})}"
        )
        for p in per_root:
            print(
                f"      root {p['root']}: "
                f"len={p['schedule_len']:>2}  tree_cost={p['tree_cost']:>10.0f}  "
                f"max Φ={p['max_phi']:>7} B  vio={p['n_violations']}"
            )

        if not all_violations:
            print(f"\nfixpoint reached at round {outer} (V = ∅).")
            break

        if new_subsumes == 0:
            quiescent_rounds += 1
            if quiescent_rounds >= 2:
                tot_v = sum(p["n_violations"] for p in per_root)
                blocked = "infeasible (no actionable rewrite)" if n_unactionable >= tot_v \
                          else "alias-blocked (commutativity)"
                print(f"\nstopping: {quiescent_rounds} consecutive rounds without new subsumes "
                      f"(remaining {tot_v} violations are {blocked}).")
                break
        else:
            quiescent_rounds = 0

    return history, extras


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel",  default=DEFAULT_KERNEL)
    ap.add_argument("--smax",    type=int, default=DEFAULT_SMAX)
    ap.add_argument("--sz",      type=int, default=DEFAULT_SZ)
    ap.add_argument("--rounds",  type=int, default=10)
    ap.add_argument("--max-subsumes-per-round", type=int, default=8,
                    help="cap on subsume actions emitted per outer iteration")
    ap.add_argument("--workdir", default="experiments/_flow_prune")
    ap.add_argument("--csv",     default="experiments/flow_prune_results.csv")
    ap.add_argument("--meta",    default=None,
                    help="optional path to a tensor-meta JSON (e.g. ACT PII "
                         "side-map). When provided, on-chip footprint is "
                         "computed per-PNamed using meta[name]['onchip_bytes'].")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    kernel    = (repo_root / args.kernel).resolve()
    workdir   = (repo_root / args.workdir).resolve()
    egglog    = find_egglog_bin(repo_root)

    meta = None
    if args.meta:
        meta = json.loads(Path(args.meta).read_text())

    print(f"egglog : {egglog}")
    print(f"kernel : {kernel}")
    print(f"S_max  : {args.smax} bytes   sz: {args.sz} bytes/elem   tiles: {TILE_SIZES}")
    print(f"meta   : {args.meta or '(none — synthetic 3·T²·sz model)'}")
    print(f"workdir: {workdir}\n")

    history, extras = network_flow_guided_saturation(
        egglog, kernel, workdir, args.smax, args.sz, args.rounds,
        max_subsumes_per_round=args.max_subsumes_per_round,
        meta=meta,
    )

    csv_path = (repo_root / args.csv).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(history[0].keys()) if history else []
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in history:
            w.writerow(row)
    print(f"\nwrote {csv_path}")

    if extras:
        finalp = workdir / "final.subsumes.egg"
        finalp.write_text("\n".join(extras) + "\n")
        print(f"wrote {finalp}  ({len(extras)} subsume actions)")


if __name__ == "__main__":
    main()

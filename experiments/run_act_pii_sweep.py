#!/usr/bin/env python3
"""Run flow_prune on ACT PII fixtures across S_max; write `experiments/act_pii_sweep.csv`."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
PII_DIR    = ROOT / "experiments" / "act_pii"
SWEEP_OUT  = ROOT / "experiments" / "_flow_prune_act_sweep"
CSV_OUT    = ROOT / "experiments" / "act_pii_sweep.csv"
DEFAULT_SZ = 4


SCENARIOS: list[tuple[str, int, int]] = [
    ("attention",         12_000, 2),
    ("attention",         20_000, 2),
    ("attention",         65_536, 2),
    ("block_dot",          3_500, 4),
    ("block_dot",          5_000, 4),
    ("block_dot_bf16",     1_800, 2),
    ("block_dot_bf16",     3_000, 2),
    ("block_dot_pinned",   3_000, 4),
    ("block_dot_pinned",   5_000, 4),
    ("dead_heavy",         3_000, 4),
    ("dead_heavy",         5_000, 4),
    ("multi_rule",         3_000, 4),
    ("multi_rule_full",    3_000, 4),
    ("minimal_solver",     1_000, 4),
]


def run_one(stem: str, smax: int, sz: int) -> dict:
    egg  = PII_DIR / f"{stem}.pii.egg"
    meta = PII_DIR / f"{stem}.pii.meta.json"
    workdir = SWEEP_OUT / f"{stem}_smax{smax}"
    if workdir.exists():
        shutil.rmtree(workdir)

    cmd = [
        sys.executable, str(ROOT / "experiments" / "flow_prune.py"),
        "--kernel", str(egg.relative_to(ROOT)),
        "--meta",   str(meta.relative_to(ROOT)),
        "--smax",   str(smax),
        "--sz",     str(sz),
        "--rounds", "10",
        "--workdir", str(workdir.relative_to(ROOT)),
        "--csv",    str(workdir.relative_to(ROOT) / "history.csv"),
    ]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout + "\n" + res.stderr + "\n")
        raise RuntimeError(f"flow_prune failed for {stem} smax={smax}")

    hist_path = workdir / "history.csv"
    rows = list(csv.DictReader(hist_path.open()))
    last  = rows[-1]
    first = rows[0]
    return {
        "fixture": stem,
        "smax_B": smax,
        "rounds": int(last["round"]) + 1,
        "init_max_phi_B": int(first["max_phi"]),
        "final_max_phi_B": int(last["max_phi"]),
        "init_violations": int(first["n_violations"]),
        "final_violations": int(last["n_violations"]),
        "final_unactionable": int(last.get("n_unactionable", 0)),
        "total_subsumes": int(last["total_subsumes"]),
        "outcome": ("feasible (V=∅)" if int(last["n_violations"]) == 0
                    else "infeasible-no-rewrite"
                         if int(last.get("n_unactionable", 0)) >= int(last["n_violations"])
                         else "alias-blocked"),
    }


def main() -> None:
    SWEEP_OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for stem, smax, sz in SCENARIOS:
        r = run_one(stem, smax, sz)
        results.append(r)
        print(f"  {stem:<20} smax={smax:>6}  init Φ={r['init_max_phi_B']:>5} final Φ={r['final_max_phi_B']:>5}  "
              f"init/final V = {r['init_violations']:>2}/{r['final_violations']:<2}  "
              f"subsumes={r['total_subsumes']:<2}  rounds={r['rounds']:<2}  → {r['outcome']}")

    fields = list(results[0].keys())
    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nwrote {CSV_OUT}")


if __name__ == "__main__":
    main()

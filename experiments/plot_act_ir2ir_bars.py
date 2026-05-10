#!/usr/bin/env python3
"""Stacked-bar chart of `TiledDot` survival vs round for the act_ir2ir egglog
demo, with two side-by-side subplots — one per S_max scenario.

Counts are over **non-subsumed `TiledDot` e-nodes still alive in the e-graph
after each round**, grouped by tile size.  Mirrors the look-and-feel of
`plot_tile_bars.py` (same color palette, same Overleaf-2-column footprint).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/.mplcfg")
sys.setrecursionlimit(50000)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# (subplot title, workdir relative to repo root)
SCENARIOS: list[tuple[str, str]] = [
    ("S_max = 65,536 B  (1 prune step)",  "experiments/_flow_prune_act_ir2ir"),
    ("S_max = 20,000 B  (2 prune steps)", "experiments/_flow_prune_act_ir2ir_tight"),
]

TILES  = (128, 64, 32)
COLORS = {
    128: "#D55E00",   # vermillion (most aggressive → pruned first)
     64: "#E69F00",   # orange
     32: "#009E73",   # bluish-green (final fallback)
}


def per_round_counts(workdir: Path) -> list[dict]:
    """Count surviving (non-subsumed) `TiledDot` e-nodes per round, by tile size."""
    rounds = sorted(glob.glob(str(workdir / "round*.json")))
    out = []
    for jp in rounds:
        rd = int(jp.rsplit("round", 1)[1].split(".")[0])
        eg = json.loads(Path(jp).read_text())
        nodes = eg["nodes"]
        c: Counter = Counter()
        for nid, n in nodes.items():
            if n.get("subsumed", False):
                continue
            if n["op"] != "TiledDot":
                continue
            try:
                T = int(nodes[n["children"][2]]["op"])
            except (KeyError, ValueError):
                continue
            if T in TILES:
                c[T] += 1
        out.append({"round": rd, **{T: c.get(T, 0) for T in TILES}})
    return out


def make_plotly(data: list[tuple[str, list[dict]]]) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=len(data),
        subplot_titles=[t for t, _ in data],
        horizontal_spacing=0.10,
    )
    for col, (_, rows) in enumerate(data, start=1):
        x = [str(r["round"]) for r in rows]
        for T in TILES:
            y = [r[T] for r in rows]
            fig.add_trace(
                go.Bar(
                    x=x, y=y,
                    name=f"T={T}",
                    marker=dict(color=COLORS[T], line=dict(width=0)),
                    legendgroup=f"T={T}",
                    showlegend=(col == 1),
                ),
                row=1, col=col,
            )
        fig.update_xaxes(title_text="round", row=1, col=col)
        fig.update_yaxes(title_text="surviving TiledDot count" if col == 1 else None,
                         row=1, col=col, rangemode="tozero")
    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        font=dict(family="Latin Modern Roman, Times New Roman, serif", size=14),
        height=340, width=860,
        margin=dict(l=70, r=20, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.10,
                    xanchor="right", x=1.0,
                    bgcolor="rgba(0,0,0,0)"),
        bargap=0.22,
    )
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=14, family="serif")
    return fig


def make_matplotlib(data: list[tuple[str, list[dict]]], out_png: Path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, len(data), figsize=(7.0, 2.7), sharey=False)
    for ax, (title, rows) in zip(axes, data):
        x = [r["round"] for r in rows]
        bottom = [0] * len(x)
        for T in TILES:
            y = [r[T] for r in rows]
            ax.bar(x, y, bottom=bottom, color=COLORS[T],
                   edgecolor="white", linewidth=0.6, label=f"T={T}")
            bottom = [b + v for b, v in zip(bottom, y)]
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xlabel("round", fontsize=10)
        ax.set_xticks(x)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("surviving TiledDot count", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=len(TILES), frameon=False, fontsize=10,
               handlelength=1.2, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_png}")


def main():
    data = [(title, per_round_counts(ROOT / wd)) for title, wd in SCENARIOS]

    out_html = ROOT / "experiments" / "act_ir2ir_tile_bars.html"
    fig = make_plotly(data)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"wrote {out_html}")

    out_png = ROOT / "experiments" / "act_ir2ir_tile_bars.png"
    make_matplotlib(data, out_png)

    print("\nper-round counts:")
    for title, rows in data:
        print(f"  {title}")
        for r in rows:
            print(f"    round {r['round']}: " +
                  ", ".join(f"T={T}:{r[T]}" for T in TILES))


if __name__ == "__main__":
    main()

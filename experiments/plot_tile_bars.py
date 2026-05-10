#!/usr/bin/env python3
"""Stacked-bar chart of TiledHDot tile-size selection per round, sized for an
Overleaf 2-column figure (\\figure*).  Renders both an interactive Plotly HTML
and a publication-ready PNG via matplotlib (no headless-Chrome dependency)."""

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
import flow_prune as fp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODELS = ["mlp", "attention", "transformer8"]
TILES  = (128, 64, 32)
COLORS = {
    128: "#D55E00",   # vermillion (most aggressive → gets pruned away)
     64: "#E69F00",   # orange
     32: "#009E73",   # bluish green (final fallback)
}


def per_round_counts(model: str) -> list[dict]:
    rounds = sorted(glob.glob(str(ROOT / f"experiments/_flow_prune_{model}/round*.json")))
    out = []
    for jp in rounds:
        rd = int(jp.rsplit("round", 1)[1].split(".")[0])
        eg = json.loads(Path(jp).read_text())
        nodes = eg["nodes"]
        roots = fp.find_root_eclasses(eg)
        sel: set[str] = set()
        for r in roots:
            choice, _ = fp.greedy_extract(eg, r)
            if choice.get(r) is None:
                continue
            root_nid, kids = fp.realize_dag(r, choice, nodes)
            sched = fp.dfs_post_order(root_nid, kids)
            sel |= set(sched)
        c = Counter()
        for nid in sel:
            n = nodes[nid]
            T = fp.tile_size_of(n, nodes)
            if n["op"] == "TiledHDot":
                c[T] += 1
        out.append({"round": rd, **{T: c.get(T, 0) for T in TILES}})
    return out


def make_plotly(data: dict[str, list[dict]]) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=len(MODELS),
        subplot_titles=MODELS,
        horizontal_spacing=0.07,
    )
    for col, model in enumerate(MODELS, start=1):
        rows = data[model]
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
        fig.update_yaxes(title_text="TiledHDot count" if col == 1 else None,
                         row=1, col=col, rangemode="tozero")
    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        font=dict(family="Latin Modern Roman, Times New Roman, serif", size=14),
        height=320, width=1000,
        margin=dict(l=60, r=20, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.10,
                    xanchor="right", x=1.0,
                    bgcolor="rgba(0,0,0,0)"),
        bargap=0.18,
    )
    for ann in fig["layout"]["annotations"]:                    # subplot titles
        ann["font"] = dict(size=15, family="serif")
    return fig


def make_matplotlib(data: dict[str, list[dict]], out_png: Path):
    """Same chart layout, but rendered via matplotlib for crisp PNG output."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.0, 2.6), sharey=False)
    for ax, model in zip(axes, MODELS):
        rows = data[model]
        x = [r["round"] for r in rows]
        bottom = [0] * len(x)
        for T in TILES:
            y = [r[T] for r in rows]
            ax.bar(x, y, bottom=bottom, color=COLORS[T],
                   edgecolor="white", linewidth=0.6, label=f"T={T}")
            bottom = [b + v for b, v in zip(bottom, y)]
        ax.set_title(model, fontsize=12, pad=6)
        ax.set_xlabel("round", fontsize=10)
        ax.set_xticks(x)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("TiledHDot count", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=len(TILES), frameon=False, fontsize=10,
               handlelength=1.2, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_png}")


def main():
    data = {m: per_round_counts(m) for m in MODELS}

    fig = make_plotly(data)
    out_html = ROOT / "experiments" / "tile_bars.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"wrote {out_html}")

    out_png = ROOT / "experiments" / "tile_bars.png"
    make_matplotlib(data, out_png)


if __name__ == "__main__":
    main()

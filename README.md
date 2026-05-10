# Network-Flow-Guided E-Graph Saturation — Artifact

This directory contains the implementation and reproduction scripts for the
saturate-analyse-prune driver described in the paper. The driver wraps the
upstream [`egglog`](https://github.com/egraphs-good/egglog) rewrite engine in a
Python outer loop that interleaves saturation, network-flow analysis, and
selective `(subsume ...)` actions to enforce on-chip memory budgets.

## 1. Prerequisites

Tested on macOS 14 (arm64) and Ubuntu 22.04. You need:

* Python ≥ 3.9
* Rust toolchain (`cargo`) — only used to install `egglog`
* (optional) JAX + Flax, only if you want to regenerate the HLO dumps

## 2. Setup

```bash
# 1) install the upstream egglog binary
cargo install --git https://github.com/egraphs-good/egglog --bin egglog

# 2) tell our driver where to find it
export EGGLOG_BIN="$(which egglog)"

# 3) install Python deps
pip install -r requirements.txt
```

## 3. File layout

```
experiments/
├── flow_prune.py            # core driver (Algorithm 1)
│
├── hlo_to_egglog.py         # Exp 1 frontend: HLO text  →  .egg
├── adapt_jax_egg.py         # Exp 1: inject tile rewrites  →  .tiled.egg
├── jax_models.py            # (optional) JAX/Flax sources of the workloads
├── scale_hlo.py             # (optional) batch  jax.jit().lower() → .hlo  → .egg
│
├── jax_hlo/                 # pre-generated XLA HLO text (so JAX is optional)
│   └── {mlp,attention,transformer8}.hlo
├── jax_egglog/              # pre-generated egglog (raw + tile-injected)
│   └── {mlp,attention,transformer8}{,.tiled}.egg
│
├── kernels/
│   ├── act_ir2ir.egg        # Exp 2: ACT IR2IR subset + tile rewrites (matmul)
│   └── TASO-Sensat.egg      # (optional) early synthetic kernel
│
├── plot_tile_bars.py        # Exp 1 figure
├── plot_act_ir2ir_bars.py   # Exp 2 figure
└── run_act_pii_sweep.py     # (optional) ACT PII feasibility sweep
```

## 4. Reproducing Experiment 1 (HLO + Sensat + tile)

```bash
# pick one of: mlp, attention, transformer8
MODEL=mlp

python3 experiments/flow_prune.py \
    --kernel  experiments/jax_egglog/${MODEL}.tiled.egg \
    --smax    65536 \
    --sz      4 \
    --rounds  10 \
    --workdir experiments/_flow_prune_${MODEL} \
    --csv     experiments/flow_prune_${MODEL}.csv

# After running for all three models, render the figure:
python3 experiments/plot_tile_bars.py
# → experiments/tile_bars.png   (and tile_bars.html for the interactive version)
```

To regenerate the `.tiled.egg` files from scratch (only needed if you want
to re-run the HLO frontend):

```bash
python3 experiments/hlo_to_egglog.py experiments/jax_hlo/${MODEL}.hlo \
        > experiments/jax_egglog/${MODEL}.egg
python3 experiments/adapt_jax_egg.py experiments/jax_egglog/${MODEL}.egg
```

## 5. Reproducing Experiment 2 (ACT IR2IR subset + tile, matmul)

```bash
# scenario A: S_max = 65,536 B  (1 prune step)
python3 experiments/flow_prune.py \
    --kernel  experiments/kernels/act_ir2ir.egg \
    --smax    65536 \
    --sz      4 \
    --rounds  6 \
    --workdir experiments/_flow_prune_act_ir2ir \
    --csv     experiments/flow_prune_act_ir2ir.csv

# scenario B: S_max = 20,000 B  (2 prune steps)
python3 experiments/flow_prune.py \
    --kernel  experiments/kernels/act_ir2ir.egg \
    --smax    20000 \
    --sz      4 \
    --rounds  6 \
    --workdir experiments/_flow_prune_act_ir2ir_tight \
    --csv     experiments/flow_prune_act_ir2ir_tight.csv

# Render the side-by-side figure:
python3 experiments/plot_act_ir2ir_bars.py
# → experiments/act_ir2ir_tile_bars.png
```

## 6. What the driver prints

Each round prints one line of the form:

```
round  0 | roots= 1 | violations=  4 (unactionable=0) | max Φ= 196608 B | +subsume=1 (total 1) | |V|=32 |classes|=22
```

* `violations` — # program points where Φ(L_p) > S_max
* `unactionable` — violations whose live tensors have no tile alternative
* `max Φ` — peak on-chip footprint in this round
* `+subsume` / `total` — new subsume actions emitted, and cumulative count
* `|V|` / `|classes|` — current e-graph size

The loop terminates when `violations = 0` (true fixpoint) or two consecutive
rounds emit no new subsumes (infeasible under the current rule set).

## 7. Outputs

Per-run, the workdir contains:
```
round00.egg  round00.json  round01.egg  round01.json  ...   final.subsumes.egg
```

`final.subsumes.egg` lists all `(subsume ...)` actions chosen across the run.
The CSV summarises one row per round (rounds, violations, peak Φ, etc.).

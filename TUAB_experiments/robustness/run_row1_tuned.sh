#!/usr/bin/env bash
# reference_table_robustness_V1.md — ROW 1: TUNED MODEL (the winner of the campaign)
# Conv1D 1.35M (hidden=96, depth=4) · VICReg + spectral 0.1 + exact-corruption
# inv=1 · no scale-jitter · 20 ep · frozen probe
# Expected: per-rec BAcc/AUROC = 0.824 / 0.896, per-win = 0.755 / 0.839
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"; source env.sh 2>/dev/null || true

CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/robustness_row1_tuned}"

python -m examples.eeg.main \
    --fname examples/eeg/cfgs/train.yaml \
    meta.ckpt_dir="$CKPT_DIR" \
    model.hidden=96 \
    model.depth=4 \
    model.inv_coeff=1.0 \
    model.spectral_coeff=0.1 \
    data.aug_exact_corruption=true \
    data.aug_scale_jitter=0.0

python -m examples.eeg.eval --ckpt "$CKPT_DIR/latest.pth.tar" --level both --probe both

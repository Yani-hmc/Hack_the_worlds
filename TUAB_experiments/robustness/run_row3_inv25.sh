#!/usr/bin/env bash
# reference_table_robustness_V1.md — ROW 3: HP TUNING — invariance weight inv=25
# Tuned recipe with model.inv_coeff=25 (VICReg default) instead of our tuned inv=1
# Expected: per-rec BAcc/AUROC = 0.804 / 0.884, per-win = 0.730 / 0.813
# (-2 pp vs tuned — EEG SSL prefers WEAK view-invariance; full sweep: 1 > 5 > 10 > 25)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"; source env.sh 2>/dev/null || true

CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/robustness_row3_inv25}"

python -m examples.eeg.main \
    --fname examples/eeg/cfgs/train.yaml \
    meta.ckpt_dir="$CKPT_DIR" \
    model.hidden=96 \
    model.depth=4 \
    model.inv_coeff=25.0 \
    model.spectral_coeff=0.1 \
    data.aug_exact_corruption=true \
    data.aug_scale_jitter=0.0

python -m examples.eeg.eval --ckpt "$CKPT_DIR/latest.pth.tar" --level both --probe both

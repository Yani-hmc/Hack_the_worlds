#!/usr/bin/env bash
# reference_table_robustness_V1.md — ROW 2: ROBUSTNESS — pretrain on 25% of recordings
# Tuned recipe + data.subset_frac=0.25 (class-balanced subsampling)
# Expected: per-rec BAcc/AUROC = 0.816 / 0.895, per-win = 0.762 / 0.842
# (-0.8 pp vs tuned — encoder saturates early, NOT data-starved)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"; source env.sh 2>/dev/null || true

CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/robustness_row2_25pct}"

python -m examples.eeg.main \
    --fname examples/eeg/cfgs/train.yaml \
    meta.ckpt_dir="$CKPT_DIR" \
    model.hidden=96 \
    model.depth=4 \
    model.inv_coeff=1.0 \
    model.spectral_coeff=0.1 \
    data.aug_exact_corruption=true \
    data.aug_scale_jitter=0.0 \
    data.subset_frac=0.25

python -m examples.eeg.eval --ckpt "$CKPT_DIR/latest.pth.tar" --level both --probe both

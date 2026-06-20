#!/usr/bin/env bash
# reference_table_robustness_V1.md — ROW 4: ABLATION — remove the corruption mask
# Tuned recipe with data.aug_mask_frac=0.0 (outliers only, no time-masking)
# Expected: per-rec BAcc/AUROC = 0.796 / 0.891, per-win = 0.762 / 0.841
# (-2.8 pp vs tuned — the mask is the single biggest aug lever)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"; source env.sh 2>/dev/null || true

CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/robustness_row4_no_mask}"

python -m examples.eeg.main \
    --fname examples/eeg/cfgs/train.yaml \
    meta.ckpt_dir="$CKPT_DIR" \
    model.hidden=96 \
    model.depth=4 \
    model.inv_coeff=1.0 \
    model.spectral_coeff=0.1 \
    data.aug_exact_corruption=true \
    data.aug_scale_jitter=0.0 \
    data.aug_mask_frac=0.0

python -m examples.eeg.eval --ckpt "$CKPT_DIR/latest.pth.tar" --level both --probe both

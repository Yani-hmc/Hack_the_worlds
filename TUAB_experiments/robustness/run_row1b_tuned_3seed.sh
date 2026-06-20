#!/usr/bin/env bash
# reference_table_robustness_V1.md — ROW 1b: TUNED MODEL — 3-seed (the noise band)
# Same recipe as row 1; seeds {0, 1, 2}, report mean±std
# Expected: per-rec BAcc/AUROC = 0.812 ± .021 / 0.898 ± .002, per-win = 0.756 ± .001 / 0.836 ± .005
# (Note: per-rec carries ≈±0.02 seed noise on 276 recordings — per-win/AUROC much more stable.)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"; source env.sh 2>/dev/null || true

for SEED in 0 1 2; do
    CKPT_DIR="${ROOT}/checkpoints/robustness_row1b_seed${SEED}"
    python -m examples.eeg.main \
        --fname examples/eeg/cfgs/train.yaml \
        meta.ckpt_dir="$CKPT_DIR" \
        meta.seed=$SEED \
        model.hidden=96 \
        model.depth=4 \
        model.inv_coeff=1.0 \
        model.spectral_coeff=0.1 \
        data.aug_exact_corruption=true \
        data.aug_scale_jitter=0.0
    python -m examples.eeg.eval --ckpt "$CKPT_DIR/latest.pth.tar" --level both --probe both
done

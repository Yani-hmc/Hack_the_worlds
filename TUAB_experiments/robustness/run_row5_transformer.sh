#!/usr/bin/env bash
# reference_table_robustness_V1.md — ROW 5: ABLATION — Transformer at equal capacity ~1.3M
# Tuned recipe with model.encoder_type=transformer (ViT-style attention encoder)
# Expected: per-rec BAcc/AUROC = 0.725 / 0.804, per-win = 0.652 / 0.708
# (-10 pp vs tuned — at this scale/epoch budget the Conv wins decisively;
# the SOTA gap is pretraining corpus + schedule, NOT architecture)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"; source env.sh 2>/dev/null || true

CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/robustness_row5_transformer}"

python -m examples.eeg.main \
    --fname examples/eeg/cfgs/train.yaml \
    meta.ckpt_dir="$CKPT_DIR" \
    model.encoder_type=transformer \
    model.embed_dim=192 \
    model.tr_depth=4 \
    model.n_heads=6 \
    model.inv_coeff=1.0 \
    model.spectral_coeff=0.1 \
    data.aug_exact_corruption=true \
    data.aug_scale_jitter=0.0

python -m examples.eeg.eval --ckpt "$CKPT_DIR/latest.pth.tar" --level both --probe both

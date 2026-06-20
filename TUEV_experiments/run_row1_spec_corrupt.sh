#!/usr/bin/env bash
# reference_table_JEPA_TUEV_V1.md — ROW 1: VICReg + spectral 0.1 + corrupt  ⭐ BEST
# Conv1D 0.4M · VICReg* (inv_coeff=1) + DDSP spectral 0.1 + exact-corruption
# TUAB-pretrained, FROZEN, probed on TUEV
# Expected: TUEV BAcc = 0.425 (best — beats supervised CNN-Transformer 0.410
#                              and ST-Transformer 0.339 without seeing TUEV labels)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"; source env.sh 2>/dev/null || true

CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/tuev_row1_spec_corrupt}"

python -m examples.eeg.main \
    --fname examples/eeg/cfgs/train.yaml \
    meta.ckpt_dir="$CKPT_DIR" \
    model.inv_coeff=1.0 \
    model.spectral_coeff=0.1 \
    data.aug_exact_corruption=true

python TUEV_experiments/tuev_probe.py --ckpt "$CKPT_DIR/latest.pth.tar"

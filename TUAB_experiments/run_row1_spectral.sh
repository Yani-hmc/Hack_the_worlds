#!/usr/bin/env bash
# reference_table_JEPA_V1.md — ROW 1: VICReg + spectral 0.1
# Conv1D 0.4M · VICReg + DDSP multi-scale spectral loss (coeff 0.1) + exact-corruption · frozen
# Expected: per-rec BAcc/AUROC = 0.836 / 0.887, per-win BAcc = 0.765
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"; source env.sh 2>/dev/null || true

CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/tuab_row1_spectral}"

python -m examples.eeg.main \
    --fname examples/eeg/cfgs/train.yaml \
    meta.ckpt_dir="$CKPT_DIR" \
    data.aug_exact_corruption=true \
    model.spectral_coeff=0.1

python -m examples.eeg.eval --ckpt "$CKPT_DIR/latest.pth.tar" --level both --probe both

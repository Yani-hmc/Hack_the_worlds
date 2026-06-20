#!/usr/bin/env bash
# reference_table_JEPA_V1.md — ROW 2: SIGReg
# Conv1D 0.4M · SIGReg (LeJEPA / BCS Epps-Pulley Gaussianity test) + exact-corruption · frozen
# Expected: per-rec BAcc/AUROC = 0.825 / 0.913, per-win BAcc/AUROC = 0.775 / 0.856 (best per-win)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"; source env.sh 2>/dev/null || true

CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/tuab_row2_sigreg}"

python -m examples.eeg.main \
    --fname examples/eeg/cfgs/train.yaml \
    meta.ckpt_dir="$CKPT_DIR" \
    data.aug_exact_corruption=true \
    model.ssl_loss=sigreg

python -m examples.eeg.eval --ckpt "$CKPT_DIR/latest.pth.tar" --level both --probe both

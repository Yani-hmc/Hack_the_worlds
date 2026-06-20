#!/usr/bin/env bash
# reference_table_JEPA_TUEV_V1.md — ROW 3: SIGReg + corrupt
# Conv1D 0.4M · SIGReg (LeJEPA / BCS) + exact-corruption · TUAB-pretrained, FROZEN, probed on TUEV
# Expected: TUEV BAcc = 0.363
# (SIGReg ≥ VICReg on TUAB per-window, but here lags on TUEV transfer)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"; source env.sh 2>/dev/null || true

CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/tuev_row3_sigreg_corrupt}"

python -m examples.eeg.main \
    --fname examples/eeg/cfgs/train.yaml \
    meta.ckpt_dir="$CKPT_DIR" \
    model.ssl_loss=sigreg \
    data.aug_exact_corruption=true

python TUEV_experiments/tuev_probe.py --ckpt "$CKPT_DIR/latest.pth.tar"

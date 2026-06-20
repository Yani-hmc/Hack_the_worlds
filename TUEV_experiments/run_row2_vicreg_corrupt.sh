#!/usr/bin/env bash
# reference_table_JEPA_TUEV_V1.md — ROW 2: VICReg + corrupt (seed 0)
# Conv1D 0.4M · VICReg* (inv_coeff=1) + exact-corruption · TUAB-pretrained, FROZEN, probed on TUEV
# Expected: TUEV BAcc = 0.382 (corruption alone gives no lift on TUEV — different from TUAB)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"; source env.sh 2>/dev/null || true

CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/tuev_row2_vicreg_corrupt}"

python -m examples.eeg.main \
    --fname examples/eeg/cfgs/train.yaml \
    meta.ckpt_dir="$CKPT_DIR" \
    meta.seed=0 \
    model.inv_coeff=1.0 \
    data.aug_exact_corruption=true

python TUEV_experiments/tuev_probe.py --ckpt "$CKPT_DIR/latest.pth.tar"

#!/usr/bin/env bash
# reference_table_JEPA_V1.md — BASELINE
# Conv1D 0.4M · VICReg · base aug (no exact-corruption) · frozen probe
# Expected: per-rec BAcc/AUROC = 0.796 / 0.888, per-win BAcc = 0.756
#
# Notes: "base aug" = legacy contiguous time-mask (data.aug_exact_corruption=false,
# which is the default). VICReg with paper-correct (inv, var, cov) = (25, 25, 1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"; source env.sh 2>/dev/null || true

CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/tuab_baseline}"

python -m examples.eeg.main \
    --fname examples/eeg/cfgs/train.yaml \
    meta.ckpt_dir="$CKPT_DIR"

python -m examples.eeg.eval --ckpt "$CKPT_DIR/latest.pth.tar" --level both --probe both

#!/usr/bin/env bash
# reference_table_JEPA_TUEV_V1.md — BASELINE
# Conv1D 0.4M · VICReg · base aug (no corruption) · TUAB-pretrained, FROZEN, probed on TUEV
# Expected: TUEV BAcc = 0.381 (chance = 0.167; random-encoder floor = 0.337)
#
# Two stages: (1) train SSL encoder on TUAB, (2) run TUEV 6-class probe (frozen encoder).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"; source env.sh 2>/dev/null || true

CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/tuev_baseline}"

# (1) train SSL encoder on TUAB
python -m examples.eeg.main \
    --fname examples/eeg/cfgs/train.yaml \
    meta.ckpt_dir="$CKPT_DIR"

# (2) TUEV 6-class probe (frozen encoder)
python TUEV_experiments/tuev_probe.py --ckpt "$CKPT_DIR/latest.pth.tar"

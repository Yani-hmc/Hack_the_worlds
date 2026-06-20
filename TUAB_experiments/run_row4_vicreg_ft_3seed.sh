#!/usr/bin/env bash
# reference_table_JEPA_V1.md — ROW 4: VICReg + FT (end-to-end fine-tuning, 3-seed)
# Conv1D 0.4M · VICReg pretrained (row 3) · then FT entire encoder + head on TUAB labels
# Expected: per-rec BAcc/AUROC = 0.812 ± .004 / 0.908 ± .006
#
# Requires: a pretrained row-3 checkpoint (run_row3_vicreg_3seed.sh first).
# Uses examples/eeg/cfgs/finetune.yaml; finetune.py loads encoder via --init.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"; source env.sh 2>/dev/null || true

for SEED in 0 1 2; do
    INIT_CKPT="${ROOT}/checkpoints/tuab_row3_vicreg_seed${SEED}/latest.pth.tar"
    FT_DIR="${ROOT}/checkpoints/tuab_row4_ft_seed${SEED}"
    python -m examples.eeg.finetune \
        --fname examples/eeg/cfgs/finetune.yaml \
        --init "$INIT_CKPT" \
        meta.ckpt_dir="$FT_DIR" \
        meta.seed=$SEED
done

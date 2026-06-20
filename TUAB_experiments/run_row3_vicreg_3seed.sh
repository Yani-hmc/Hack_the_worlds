#!/usr/bin/env bash
# reference_table_JEPA_V1.md — ROW 3: VICReg (3-seed)
# Conv1D 0.4M · VICReg + exact-corruption · frozen · seeds {0, 1, 2}
# Expected: per-rec BAcc/AUROC = 0.819 ± .004 / 0.900 ± .006, per-win = 0.770 / 0.848
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"; source env.sh 2>/dev/null || true

for SEED in 0 1 2; do
    CKPT_DIR="${ROOT}/checkpoints/tuab_row3_vicreg_seed${SEED}"
    python -m examples.eeg.main \
        --fname examples/eeg/cfgs/train.yaml \
        meta.ckpt_dir="$CKPT_DIR" \
        meta.seed=$SEED \
        data.aug_exact_corruption=true
    python -m examples.eeg.eval --ckpt "$CKPT_DIR/latest.pth.tar" --level both --probe both
done

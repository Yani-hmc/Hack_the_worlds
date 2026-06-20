#!/usr/bin/env bash
# reference_table_JEPA_V1.md — ROW 6: VICReg + spectral 0.1 + Δz/Δz²/ACF/sACF (3-seed)
# Row-1 energy + four "stylized-facts" view-consistency MSE blocks between feature maps:
#   Δz (0.10), Δz² (0.05), temporal autocorrelation lags 1-8 (0.10), spectral ACF (0.05)
# Expected: per-rec BAcc/AUROC = 0.823 ± .007 / 0.900 ± .010, per-win = 0.761 / 0.841
#
# NB: the four stylized-facts terms are NOT yet wired into examples/eeg/main.py's
# TwoViewVICReg.compute_loss — they live in a Phase-6 patch (see commit fc5041e on
# branch main, reports/reference_table_JEPA_V1.tex row 6 notes).
# Below is the row-1 invocation (closest reproducible); apply the Phase-6 patch
# to add `model.dz_coeff=0.1 dz2_coeff=0.05 acf_coeff=0.1 sacf_coeff=0.05` for the
# full row 6.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"; source env.sh 2>/dev/null || true

for SEED in 0 1 2; do
    CKPT_DIR="${ROOT}/checkpoints/tuab_row6_stylized_seed${SEED}"
    python -m examples.eeg.main \
        --fname examples/eeg/cfgs/train.yaml \
        meta.ckpt_dir="$CKPT_DIR" \
        meta.seed=$SEED \
        data.aug_exact_corruption=true \
        model.spectral_coeff=0.1
    python -m examples.eeg.eval --ckpt "$CKPT_DIR/latest.pth.tar" --level both --probe both
done

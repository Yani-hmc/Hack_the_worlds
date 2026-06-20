#!/bin/bash
#SBATCH --job-name=biotzoo
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=biotzoo_%j.out
# Reproduce the LaBraM-Table-1 baseline panel from the OFFICIAL BIOT repo
# (github.com/ycq091044/BIOT) on OUR TUAB_PREPROCESSED pipeline. One GPU
# allocation loops over all from-scratch models (BIOT itself already run via
# run_biot.sbatch). Override the set with:  MODELS="STTransformer CNNTransformer" sbatch ...
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"

BIOT_REPO="$WORK/external/BIOT"
if [ ! -d "$BIOT_REPO/model" ]; then
    echo "!! BIOT repo not found at $BIOT_REPO -- run setup_externals.sh first"; exit 1
fi

# 19 channels = same montage our JEPA uses (these are FROM-SCRATCH, no 18-ch
# pretrained-weight constraint), so it is a true apples-to-apples panel.
N_CH=19
MODELS="${MODELS:-SPaRCNet ContraWR CNNTransformer FFCL STTransformer}"
EPOCHS="${EPOCHS:-8}"          # reduced from 20 to keep all-windows mode within walltime
N_WINDOWS="${N_WINDOWS:--1}"   # -1 = ALL non-overlapping windows (literature protocol)
echo "EPOCHS=$EPOCHS  N_WINDOWS=$N_WINDOWS  MODELS=$MODELS"

for M in $MODELS; do
    echo ""
    echo "############################################################"
    echo "### $M  | n_channels=$N_CH | $(date)"
    echo "############################################################"
    PYTHONPATH="$BIOT_REPO:$PYTHONPATH" \
        uv run python -m baselines.biot.adapter_baselines \
            --model "$M" --epochs $EPOCHS --batch-size 128 --lr 1e-3 \
            --n-channels $N_CH --n-windows $N_WINDOWS || echo "!! $M FAILED (continuing)"
done
echo ""
echo "=== DONE biotzoo: $MODELS ==="

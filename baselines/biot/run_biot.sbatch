#!/bin/bash
#SBATCH --job-name=biot
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=biot_%x_%j.out
# Run the OFFICIAL BIOT model from github.com/ycq091044/BIOT on our TUAB_PREPROCESSED
# pipeline. Two modes:
#   1) supervised from scratch (default) -- matches BIOT Tbl 4 "Vanilla BIOT" line
#   2) fine-tune from pretrained ckpt    -- set PRETRAINED=<.ckpt path>
#
# Usage:
#   sbatch baselines/biot/run_biot.sbatch
#   PRETRAINED=$WORK/external/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt \
#     sbatch baselines/biot/run_biot.sbatch
#
# Prereqs (run once on the login node):
#   bash baselines/external/setup_externals.sh
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"

BIOT_REPO="$WORK/external/BIOT"
if [ ! -d "$BIOT_REPO/model" ]; then
    echo "!! BIOT repo not found at $BIOT_REPO -- run setup_externals.sh first"
    exit 1
fi

# Slice our 19-channel data to 18 to match BIOT's pretrained 18-channel models.
N_CH=18
EXTRA_ARGS=""
if [ -n "${PRETRAINED:-}" ]; then
    if [ ! -f "$PRETRAINED" ]; then
        echo "!! PRETRAINED checkpoint not found at $PRETRAINED"
        exit 1
    fi
    EXTRA_ARGS="--pretrained-ckpt $PRETRAINED"
    TAG="pretrained"
else
    TAG="vanilla"
fi

echo "=== BIOT $TAG | n_channels=$N_CH | $(date) ==="
echo "    PRETRAINED=${PRETRAINED:-(none, supervised from scratch)}"

EPOCHS="${EPOCHS:-20}"          # train epochs (default 20; ok at N_TRAIN=16)
N_TRAIN="${N_TRAIN:-16}"        # windows/rec at TRAIN (16 = fast, organizers' default)
N_EVAL="${N_EVAL:--1}"          # windows/rec at EVAL (-1 = ALL non-overlapping, literature)
echo "    EPOCHS=$EPOCHS  N_TRAIN=$N_TRAIN  N_EVAL=$N_EVAL"
PYTHONPATH="$BIOT_REPO:$PYTHONPATH" \
    uv run python -m baselines.biot.adapter_biot \
        --epochs $EPOCHS --batch-size 128 --lr 1e-3 --n-channels $N_CH \
        --n-windows-train $N_TRAIN --n-windows-eval $N_EVAL $EXTRA_ARGS

echo "=== DONE BIOT $TAG ==="

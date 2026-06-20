#!/bin/bash
#SBATCH --job-name=labram
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=labram_%x_%j.out
# Run the OFFICIAL LaBraM-Base model (github.com/935963004/LaBraM) from the
# authors' pretrained checkpoint on OUR TUAB_PREPROCESSED pipeline.
#   MODE=finetune (default) -- end-to-end fine-tune, comparable to LaBraM's 0.814 paper number
#   MODE=frozen             -- freeze backbone, linear head only (comparable to our JEPA frozen probe)
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"

L="$WORK/external/LaBraM"
if [ ! -f "$L/checkpoints/labram-base.pth" ]; then
    echo "!! labram-base.pth not found at $L/checkpoints -- clone the repo first"; exit 1
fi

# LaBraM needs timm==0.4.12 (old layers/registry import paths) + einops. `uv run`
# re-syncs the locked env each call, so a plain `uv pip install` would be reverted;
# `--with` injects the deps for THIS run and persists through the python invocation.
MODE="${MODE:-finetune}"
EPOCHS="${EPOCHS:-20}"        # train epochs (default 20; ok at N_TRAIN=16)
N_TRAIN="${N_TRAIN:-16}"      # windows/rec at TRAIN (16 = fast)
N_EVAL="${N_EVAL:--1}"        # windows/rec at EVAL (-1 = ALL non-overlapping, literature)
echo "=== LaBraM-Base | mode=$MODE | EPOCHS=$EPOCHS N_TRAIN=$N_TRAIN N_EVAL=$N_EVAL | $(date) ==="
PYTHONPATH="$L:$PYTHONPATH" \
    uv run --with "timm==0.4.12" --with einops \
    python -m baselines.labram.adapter_labram \
        --ckpt "$L/checkpoints/labram-base.pth" \
        --mode "$MODE" --epochs $EPOCHS --batch-size 64 --lr 1e-4 --n-channels 19 \
        --n-windows-train $N_TRAIN --n-windows-eval $N_EVAL
echo "=== DONE LaBraM $MODE ==="

#!/bin/bash
#SBATCH --job-name=eeg
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=eeg_%x_%j.out
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"
TAG="${TAG:-base}"; CKPT="$WORK/checkpoints/eeg/$TAG"; mkdir -p "$CKPT"
echo "=== TRAIN tag=$TAG overrides: ${OVR:-(none)} ==="
uv run python _run.py fname=examples/eeg/cfgs/train.yaml meta.ckpt_dir="$CKPT" data.num_workers=16 $OVR
echo "=== EVAL tag=$TAG ==="
uv run python -m examples.eeg.eval --ckpt "$CKPT/latest.pth.tar" --probe both | tee "$CKPT/eval.txt"
echo "=== DONE tag=$TAG ==="

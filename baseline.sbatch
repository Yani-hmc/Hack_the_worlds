#!/bin/bash
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=bl_%x_%j.out
set -e
REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"; source "$REPO/env.sh"; cd "$REPO"
echo ">> $MOD $ARGS"
uv run python -m "$MOD" $ARGS

#!/bin/bash
#SBATCH --job-name=image_jepa_cifar
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
# No hardcoded --account: SLURM uses your DEFAULT account (your team allocation), so this
# works on any team. Override with: sbatch --account=<acct> run_cifar.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
# Logs go to the submit dir (your work-repo): a static #SBATCH path can't hold the
# auto-detected team segment, and the submit dir always exists on /work.
#SBATCH --output=image_jepa_cifar_%j.out
#SBATCH --error=image_jepa_cifar_%j.err

# Train Image-JEPA (ResNet18 backbone) on CIFAR-10 and report the online linear-probe
# accuracy (logged each epoch as val_acc). Mirrors the conventions in ../slurm_test.sh.
#
# Default config: examples/image_jepa/cfgs/default.yaml  (ResNet18 + VICReg, 300 epochs).
# To reproduce the paper's best (~91.02% SIGReg / BCS), override CFG below to sigreg.yaml.
#
# Usage:
#   sbatch examples/image_jepa/run_cifar.sbatch                       # default VICReg config
#   CFG=examples/image_jepa/cfgs/sigreg.yaml sbatch examples/image_jepa/run_cifar.sbatch
#   EXTRA="optim.epochs=5 logging.log_wandb=false" sbatch examples/image_jepa/run_cifar.sbatch   # smoke test

set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"

# Config file and any extra dot-notation overrides (space-separated).
CFG="${CFG:-examples/image_jepa/cfgs/default.yaml}"
EXTRA="${EXTRA:-}"

echo "=== Host: $(hostname) | Arch: $ARCH | Date: $(date) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a') ==="
echo "=== venv: $UV_PROJECT_ENVIRONMENT ==="
echo "=== EBJEPA_DSETS: $EBJEPA_DSETS ==="
echo "=== Config: $CFG | Extra overrides: $EXTRA ==="

module load python312
echo "=== Python: $(python3 --version) ==="

# Install uv for this arch if needed (compute nodes are aarch64).
if ! uv --version &>/dev/null; then
    echo ">>> Installing uv for $ARCH..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi
echo "=== uv: $(uv --version) ==="

echo ">>> uv sync..."
uv sync --dev --project "$REPO"

echo ">>> Training Image-JEPA on CIFAR-10..."
uv run --project "$REPO" python -m examples.image_jepa.main --fname "$CFG" $EXTRA

echo "=== Done ==="

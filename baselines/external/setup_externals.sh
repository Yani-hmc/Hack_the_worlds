#!/usr/bin/env bash
# Clone the 4 accessible baseline repos to $WORK/external/ and install their
# python deps into the project venv. Idempotent: existing clones are reused.
#
# Run on a Dalia LOGIN node (compute nodes have no internet).
#   bash baselines/external/setup_externals.sh
#
# Status table for each repo lives in baselines/external/README.md.
set -e

# Source env.sh for $WORK / paths if not already in scope
if [ -z "$WORK" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/../../env.sh"
fi

EXT="$WORK/external"
mkdir -p "$EXT"
echo "=== Cloning into $EXT ==="

clone_one() {
    local repo="$1"; local name="$2"
    if [ -d "$EXT/$name" ]; then
        echo ">>> $name already cloned, skipping"
    else
        echo ">>> cloning $repo"
        git clone --depth 1 "https://github.com/$repo.git" "$EXT/$name"
    fi
}

clone_one ycq091044/BIOT              BIOT
clone_one BINE022/EEGPT               EEGPT
clone_one SPOClab-ca/BENDR            BENDR
clone_one pulp-bio/BioFoundation      BioFoundation

echo ""
echo "=== Installing python deps for BIOT (the one we actually run) ==="
# `linear_attention_transformer` is the only non-trivial BIOT dep (rest already in pyproject)
uv pip install linear_attention_transformer einops pytorch_lightning

echo ""
echo "=== Summary ==="
ls -la "$EXT"
echo ""
echo "Ready to run: sbatch baselines/biot/run_biot.sbatch"
echo "BIOT pretrained: PRETRAINED=\$WORK/external/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt sbatch baselines/biot/run_biot.sbatch"
echo ""
echo "EEGPT / BENDR / FEMBA: see baselines/external/README.md for status (not run by default)."

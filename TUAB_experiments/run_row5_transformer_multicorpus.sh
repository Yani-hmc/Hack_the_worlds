#!/usr/bin/env bash
# reference_table_JEPA_V1.md — ROW 5: Transformer (multi-corpus) 3.65M
# ViT-style attention encoder, ~3.65M params, VICReg + exact-corruption, frozen
# Pretrained on TUAB+TUEV+TUSZ+TUEP (4× SSL data); evaluated on TUAB only
# Expected: per-rec BAcc/AUROC = 0.798 / 0.872, per-win BAcc = 0.738
#
# Multicorpus = symlink/cat the four corpora into a shared SSL root before training,
# then point data.data_root at that union. Replace MULTI_ROOT below as needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"; source env.sh 2>/dev/null || true

MULTI_ROOT="${MULTI_ROOT:-/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV-TUSZ-TUEP_PREPROCESSED}"
CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/tuab_row5_transformer_multi}"

python -m examples.eeg.main \
    --fname examples/eeg/cfgs/train.yaml \
    meta.ckpt_dir="$CKPT_DIR" \
    data.data_root="$MULTI_ROOT" \
    data.aug_exact_corruption=true \
    model.encoder_type=transformer

# Eval on TUAB only (re-point data_root via OmegaConf override would need a separate eval cfg;
# eval.py reads the ckpt's saved cfg — set TUAB-only path here at eval time)
python -m examples.eeg.eval --ckpt "$CKPT_DIR/latest.pth.tar" --level both --probe both

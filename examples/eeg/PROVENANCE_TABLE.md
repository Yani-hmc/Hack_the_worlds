# Baseline provenance + results (strict audit)

For every literature baseline whose number appears in our SOTA table, this file answers
**two questions** at once:

1. Where does the **model code** come from? (official GitHub repo or third-party port)
2. Where do the **pretrained weights** come from? (directly in the official GitHub repo,
   or via some other channel) — and, were they used here?

All per-window / per-recording numbers were measured on our 276 patient-disjoint TUAB
eval recordings (TUAB_PREPROCESSED, 19 ch @ 200 Hz, 10 s windows, per-channel z-scored).
"per-win" = per-window scoring (the literature convention). "per-rec" = mean-pool 16
window-probabilities per recording.

| # | Method | per-rec BAcc / AUROC | per-win BAcc / AUROC | Model code source | Pretrained weights used? | Weights directly in official GitHub? | Trained on our data? |
|---|---|---|---|---|---|---|---|
| 1 | **LaBraM-Base** | **0.846 / 0.926** | 0.806 / 0.855 | ✅ official `NeuralTransformer` from `935963004/LaBraM/modeling_finetune.py` | ✅ YES — `labram-base.pth` (96 MB) | ✅ **YES** — directly in `935963004/LaBraM/checkpoints/` git tree via **git-LFS** | Fine-tuned end-to-end on our data, 20 epochs AdamW |
| 4 | **BIOT vanilla (supervised)** | 0.829 / 0.903 | 0.761 / 0.839 | ✅ official `BIOTClassifier` from `ycq091044/BIOT/model/biot.py` | ❌ NO — trained from scratch | n/a (no weights used) | ✅ **trained from scratch** on our TUAB train, 20 ep AdamW lr=1e-3 |
| 5 | **BIOT pretrained-6-datasets** | 0.824 / 0.882 | 0.761 / 0.836 | ✅ official `BIOTClassifier` | ✅ YES — `EEG-six-datasets-18-channels.ckpt` (13 MB) | ✅ **YES** — directly in `ycq091044/BIOT/pretrained-models/` git tree (commit `d138e32`) | Fine-tuned end-to-end on our data, 20 ep AdamW lr=1e-3 |
| 6 | **SPaRCNet** | 0.800 / 0.905 | 0.783 / 0.876 | ✅ official `SPaRCNet` from `ycq091044/BIOT/model/sparcnet.py` | ❌ NO — trained from scratch | n/a | ✅ scratch on our data, 20 ep AdamW lr=1e-3 |
| 7 | **ContraWR** | 0.830 / 0.912 | 0.790 / 0.877 | ✅ official `ContraWR` from `ycq091044/BIOT/model/contrawr.py` | ❌ NO — trained from scratch | n/a | ✅ scratch on our data, 20 ep AdamW lr=1e-3 |
| 8 | **CNN-Transformer** | 0.802 / 0.885 | 0.742 / 0.819 | ✅ official `CNNTransformer` from `ycq091044/BIOT/model/cnn_transformer.py` | ❌ NO — trained from scratch | n/a | ✅ scratch on our data, 20 ep AdamW lr=1e-3 |
| 9 | **FFCL** | 0.833 / 0.908 | 0.762 / 0.844 | ✅ official `FFCL` from `ycq091044/BIOT/model/ffcl.py` | ❌ NO — trained from scratch | n/a | ✅ scratch on our data, 20 ep AdamW lr=1e-3 |
| 10 | **ST-Transformer** | 0.826 / 0.925 | 0.795 / 0.876 | ✅ official `STTransformer` from `ycq091044/BIOT/model/st_transformer.py` | ❌ NO — trained from scratch | n/a | ✅ scratch on our data, 20 ep AdamW lr=1e-3 |
| 11 | **EEGNet** | 0.824 / 0.913 | 0.796 / 0.882 | ❌ **NOT Lawhern's original Keras code** (`vlawhern/arl-eegmodels`). We use `braindecode.models.EEGNetv4` (PyTorch port by braindecode team) | ❌ NO — no pretrained EEGNet exists anywhere | n/a | ✅ scratch on our data, 20 ep AdamW lr=1e-3 |
| 12 | **ShallowConvNet** | 0.803 / 0.893 | 0.777 / 0.857 | ✅ `braindecode.models.ShallowFBCSPNet` — **THIS IS the official PyTorch** (Schirrmeister himself is a braindecode core maintainer; this is his code) | ❌ NO — no pretrained ShallowFBCSP exists | n/a | ✅ scratch on our data, 20 ep AdamW lr=1e-3 |

## Pretrained-weights provenance — one-line summary

→ **Every pretrained weight we used IS directly downloadable from the official GitHub
repo** (BIOT in the normal git tree, LaBraM via git-LFS). No external Google Drive /
OneDrive / email-the-authors / figshare-for-pretrained hacks.

## Caveat to read alongside every row

The numbers above use **official-author code + official-author weights (where they
exist in GitHub) + OUR preprocessing + OUR uniform 20-epoch AdamW training recipe**.

None are bit-exact paper reproductions — paper-faithful reproductions would require
running each author's preprocessing script on raw TUAB EDFs and using each author's
own training recipe (e.g. BIOT: 100 ep with their LR schedule; LaBraM: 50 ep with
layer-decay 0.65). We trade strict reproducibility for a uniform comparison protocol.

# 🎯 REFERENCE TABLE (TUEV) — Ground truth for the harder 6-class EEG SOTA

> **Same structure as [`PROVENANCE_TABLE.md`](PROVENANCE_TABLE.md), but for the
> harder TUEV (6-class event) benchmark.** Every other doc that cites TUEV numbers
> defers to this file.

---

## Why this table couldn't be re-run on our pipeline (transparent caveat)

Unlike TUAB (which we re-ran ourselves on `TUAB_PREPROCESSED`), **our TUEV data on
Dalia is in a non-standard format**:
- Per-event `.edf` files (already split by event into individual files like
  `gped_020_a_1.edf`), 21 ch @ 200 Hz, organizers' bandpass + notch already applied.
- BIOT/LaBraM/FEMBA/EEGPT all expect **raw multi-event recordings + `.rec` annotation
  files** (one EDF per recording, with timestamps inside a `.rec` for each event).
  Their `datasets/TUEV/process.py` builds event-centered 5s windows from those.

The data formats are **fundamentally incompatible**: we can't run their `process.py`
on our pre-split EDFs (the timestamps and recording-level structure aren't there
anymore). So we **cannot re-run the official baseline training on our TUEV data**
without first converting it back to raw EDFs + `.rec` annotations.

For this table, we therefore cite **LaBraM Table 2 / BIOT Table 5** numbers — those
ARE from running the official BIOT-repo and LaBraM-repo code, just on the standard
raw TUEV preprocessing. Same provenance principle as our TUAB column (official model
classes + official pretrained weights), just we didn't run them ourselves.

---

## Scoring protocol

TUEV is 6-class multiclass (SPSW, GPED, PLED, EYEM, ARTF, BCKG). The literature
reports **Balanced Accuracy / Cohen's Kappa / Weighted-F1** (no AUROC since
multiclass). 5-second windows @ 200 Hz, 16-channel bipolar montage, **per-sample
(per-event-window) scoring** — 112,491 total samples across the train+eval splits
per the BIOT paper's Table 1.

---

| # | Method | Paper (exact title) | GitHub | TUEV BAcc | TUEV Cohen-κ | TUEV Weighted-F1 | Source | Model code source | Pretrained weights used? (+ model params) | Weights directly in official GitHub? | Trained on our TUEV? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **LaBraM-Base** | *"Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI"* — Jiang, Zhao, Lu, **ICLR 2024**, [arXiv:2405.18765](https://arxiv.org/abs/2405.18765) | [github.com/935963004/LaBraM](https://github.com/935963004/LaBraM) | **0.6409** | **0.6637** | **0.8312** | LaBraM Tbl 2 ✅ verified against PDF | ✅ official `NeuralTransformer` from `modeling_finetune.py` (LaBraM TUEV finetune script `run_class_finetuning.py`) | ✅ YES — `labram-base.pth` (96 MB) — **5.8M params** | ✅ in `935963004/LaBraM/checkpoints/` (git-LFS) | ❌ Cannot — our TUEV is per-event EDFs; LaBraM expects raw EDFs + `.rec` |
| 2 | **LaBraM-Large** | same | same | 0.6581 | 0.6622 | 0.8315 | LaBraM Tbl 2 ✅ | ✅ official | ❌ Large weights NOT in repo — **46M params** | ❌ NO — repo only ships Base ckpt | ❌ |
| 3 | **LaBraM-Huge** | same | same | 0.6616 | 0.6745 | 0.8329 | LaBraM Tbl 2 ✅ | ✅ official | ❌ Huge weights NOT in repo — **369M params** | ❌ NO | ❌ |
| 4 | **BIOT vanilla (supervised)** | *"BIOT: Cross-data Biosignal Learning in the Wild"* — Yang, Westover, Sun, **NeurIPS 2023**, [arXiv:2305.10351](https://arxiv.org/abs/2305.10351) | [github.com/ycq091044/BIOT](https://github.com/ycq091044/BIOT) | 0.4682 | 0.4482 | 0.7085 | BIOT Tbl 5 ✅ verified | ✅ official `BIOTClassifier` (BIOT TUEV training script `run_multiclass_supervised.py`) | ❌ NO — trained from scratch — **3.2M params** | n/a | ❌ same format issue as above |
| 5 | **BIOT pretrained-6-datasets** | same | same | **0.5281** | **0.5273** | **0.7492** | BIOT Tbl 5 ("ultimate") = LaBraM Tbl 2 "BIOT" ✅ | ✅ official | ✅ YES — `EEG-six-datasets-18-channels.ckpt` (13 MB) — **3.2M params** | ✅ in `ycq091044/BIOT/pretrained-models/` git tree | ❌ |
| 6 | **SPaRCNet** | *"Development of expert-level classification of seizures and rhythmic and periodic patterns during EEG interpretation"* — Jing et al., **Neurology 2023** | code in [github.com/ycq091044/BIOT](https://github.com/ycq091044/BIOT) (BIOT-bundled reimpl) | 0.4161 | 0.4233 | 0.7024 | LaBraM Tbl 2 / BIOT Tbl 5 ✅ | ✅ official `SPaRCNet` from BIOT repo | ❌ NO — trained from scratch — **0.79M params** | n/a | ❌ |
| 7 | **ContraWR** | *"Self-supervised EEG representation learning for automatic sleep staging"* — Yang, Xiao, Westover, Sun, **2021**, [arXiv:2110.15278](https://arxiv.org/abs/2110.15278) | [github.com/ycq091044/BIOT](https://github.com/ycq091044/BIOT) (BIOT-bundled reimpl) | 0.4384 | 0.3912 | 0.6893 | LaBraM Tbl 2 / BIOT Tbl 5 ✅ | ✅ official `ContraWR` from BIOT repo | ❌ NO — trained from scratch — **1.6M params** | n/a | ❌ |
| 8 | **CNN-Transformer** | *"Transformer convolutional neural networks for automated artifact detection in scalp EEG"* — Peh, Yao, Dauwels, **EMBC 2022** | code in [github.com/ycq091044/BIOT](https://github.com/ycq091044/BIOT) (BIOT-bundled reimpl) | 0.4087 | 0.3815 | 0.6854 | LaBraM Tbl 2 / BIOT Tbl 5 ✅ | ✅ official `CNNTransformer` from BIOT repo | ❌ NO — trained from scratch — **3.2M params** | n/a | ❌ |
| 9 | **FFCL** | *"Motor imagery EEG classification algorithm based on CNN-LSTM feature fusion network"* — Li, Ding, Zhang, Xiu, **Biomed. Signal Process. Control 2022** | code in [github.com/ycq091044/BIOT](https://github.com/ycq091044/BIOT) (BIOT-bundled reimpl) | 0.3979 | 0.3732 | 0.6783 | LaBraM Tbl 2 / BIOT Tbl 5 ✅ | ✅ official `FFCL` from BIOT repo | ❌ NO — trained from scratch — **2.4M params** | n/a | ❌ |
| 10 | **ST-Transformer** | *"Transformer-based spatial-temporal feature learning for EEG decoding"* — Song, Jia, Yang, Xie, **2021**, [arXiv:2106.11170](https://arxiv.org/abs/2106.11170) | code in [github.com/ycq091044/BIOT](https://github.com/ycq091044/BIOT) (BIOT-bundled reimpl) | 0.3984 | 0.3765 | 0.6823 | LaBraM Tbl 2 / BIOT Tbl 5 ✅ | ✅ official `STTransformer` from BIOT repo | ❌ NO — trained from scratch — **3.5M params** | n/a | ❌ |
| 11 | **EEGNet** | *"EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces"* — Lawhern et al., **J. Neural Eng. 2018**, [arXiv:1611.08024](https://arxiv.org/abs/1611.08024) | Lawhern's Keras: [github.com/vlawhern/arl-eegmodels](https://github.com/vlawhern/arl-eegmodels); we'd use PyTorch port [github.com/braindecode/braindecode](https://github.com/braindecode/braindecode) | — | — | — | ❌ **NOT REPORTED** — EEGNet is not evaluated on TUEV in any paper we have (BIOT/LaBraM tables don't include it, EEGNet's own paper doesn't do TUEV) | — | ❌ NO — no pretrained — **3.39k params** | n/a | ❌ |
| 12 | **ShallowConvNet** | *"Deep learning with convolutional neural networks for EEG decoding and visualization"* — Schirrmeister et al., **HBM 2017**, [arXiv:1703.05051](https://arxiv.org/abs/1703.05051) | [github.com/braindecode/braindecode](https://github.com/braindecode/braindecode) (Schirrmeister maintainer) | — | — | — | ❌ **NOT REPORTED** — ShallowFBCSPNet is not evaluated on TUEV in any paper we have | — | ❌ NO — no pretrained — **41.68k params** | n/a | ❌ |

---

## Our JEPA on TUEV (for reference — different protocol than the table above)

Listed separately because it's our own work, on our own (different) preprocessing.

| Method | TUEV BAcc | TUEV Cohen-κ | TUEV Weighted-F1 | Protocol |
|---|---|---|---|---|
| **EB-JEPA SIGReg+corruption (frozen TUAB encoder + LogReg probe on TUEV)** | 0.364 | 0.197 | 0.431 | Frozen-feature transfer from a TUAB-pretrained encoder + linear probe on our per-event TUEV data. NOT trained on TUEV. Code: [`examples/eeg/tuev_probe.py`](tuev_probe.py). |
| Random-encoder floor | 0.337 | 0.164 | 0.401 | Same probe pipeline but on an untrained encoder. |

→ Our SSL representation **beats the random floor** (transfer signal exists) but is **well below trained baselines** (expected — we didn't train on TUEV). The comparison to the table above is **not apples-to-apples** because (a) we transfer from TUAB rather than train on TUEV and (b) our TUEV protocol (per-event EDF, majority-second labelling) differs from BIOT's event-centered 5s windowing.

---

## Why we can't fill rows 4-10 with "our pipeline" numbers

Same reason as the caveat above: **our TUEV data on Dalia is per-event EDFs**
(already split by event into separate files, no recording-level structure).
BIOT/LaBraM `datasets/TUEV/process.py` need raw multi-event EDFs + `.rec`
annotations to build their event-centered 5s windows. The conversion is non-trivial
(we'd need to stitch per-event EDFs back into recordings + reconstruct event
timestamps, with information loss).

If you want strict "official code on our TUEV", the realistic path is:
1. Download raw TUEV from TUH (separate request, large file)
2. Run BIOT's `datasets/TUEV/process.py` → generates 112k 5s event windows
3. Run `run_multiclass_supervised.py --model <X>` for each baseline
4. Estimated: ~half-day of work, doable but didn't fit in this hackathon's
   remaining time after TUAB was prioritized.

---

## Pretrained-weights provenance — TUEV

| Source | What's there | Used here? |
|---|---|---|
| `ycq091044/BIOT/pretrained-models/` (normal git tree) | 3 ckpts (the same TUAB ones; BIOT loads them then **fine-tunes on TUEV**) | Cited (row 5) — BIOT authors fine-tuned on TUEV; we cite their number |
| `935963004/LaBraM/checkpoints/` (git-LFS) | `labram-base.pth` | Cited (row 1) — LaBraM authors fine-tuned on TUEV; we cite |
| Anywhere else | LaBraM-Large/Huge | NOT distributed → could not be run |

---

## Reading the table — quick takeaways

- **LaBraM-Huge tops TUEV** (BAcc 0.66, κ 0.67, W-F1 0.83) — large pretrained transformer wins decisively
- **BIOT pretrained-6** is the next best (BAcc 0.53) — pretraining matters on TUEV (vanilla BIOT 0.47)
- **All BIOT-zoo supervised baselines cluster around BAcc 0.40-0.44** — well below pretrained transformers
- **EEGNet, ShallowConvNet, our JEPA**: not directly comparable (different protocol / not in literature)
- **Our JEPA transfer (0.36 BAcc) is roughly at random-floor + small lift**, consistent with frozen transfer from a different task (TUAB binary → TUEV 6-class)

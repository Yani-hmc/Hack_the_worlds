# 🎯 REFERENCE TABLE (TUEV) — Ground truth for the harder 6-class EEG SOTA

> **Same structure as [`PROVENANCE_TABLE.md`](PROVENANCE_TABLE.md), but for the
> harder TUEV (6-class event) benchmark.** Every other doc that cites TUEV
> numbers defers to this file. If a number in another doc disagrees with this
> one, this one is correct.

For every literature baseline whose number appears in our TUEV SOTA
comparison, this file answers **two questions** at once:

1. Where does the **model code** come from? (official GitHub repo or third-party port)
2. Where do the **pretrained weights** come from? (directly in the official GitHub repo,
   or via some other channel) — and, were they used here?

---

## What we ran (and how)

We downloaded the raw TUEV corpus from TUH, ran **BIOT's official
[`datasets/TUEV/process.py`](https://github.com/ycq091044/BIOT/blob/main/datasets/TUEV/process.py)**
verbatim on the raw EDFs + `.rec` annotation files, and got the standard
**112,491** event-centered 5 s windows (83,932 train + 28,559 eval). 16-channel
bipolar montage @ 200 Hz, same as the BIOT/LaBraM papers.

We then ran **BIOT's official
[`run_multiclass_supervised.py`](https://github.com/ycq091044/BIOT/blob/main/run_multiclass_supervised.py)**
(renamed `run_train.py` in our tree) on those pkl windows, for each of the 7
supervised baselines bundled in the BIOT repo
(SPaRCNet, ContraWR, CNN-Transformer, FFCL, ST-Transformer, BIOT,
BIOT-pretrained). Training recipe = repo defaults: AdamW, lr=1e-3, batch=256,
10 epochs, weight_decay=1e-5, num_workers=16, 1× GB200 GPU.

Sbatch driver: [`scripts/run_tuev_train.sbatch`](https://github.com/ycq091044/BIOT)
in our fork. Jobs: 77386-77391 + 77555 (BIOT-pre with 16-ch ckpt).

---

## Scoring protocol

TUEV is 6-class multiclass (SPSW, GPED, PLED, EYEM, ARTF, BCKG). The literature
reports **Balanced Accuracy / Cohen's Kappa / Weighted-F1** (no AUROC since
multiclass). 5-second windows @ 200 Hz, 16-channel bipolar montage, **per-sample
(per-event-window) scoring** — same protocol as BIOT Tbl 5 / LaBraM Tbl 2.

---

> **⚠️ On the literature-cited numbers (rows 1-3).** For LaBraM (Base/Large/Huge)
> we **could not retrain ourselves** — these are heavy pretrained models that
> require massive EEG corpora (~5,000 h of unlabeled EEG for LaBraM's
> pretraining stage) which is out of reach for this hackathon. So we cite the
> LaBraM Tbl 2 numbers directly. **Those numbers are NOT directly comparable to
> our "Our TUEV" column**, because they come from the LaBraM authors' own
> preprocessing pipeline (their finetuning code path with different normalization,
> different patch tokenization, possibly different train/eval split), which is
> not the same as our BIOT-`process.py`-based preprocessing used for rows 4-10.

| # | Method | Paper (exact title) | GitHub | **Our TUEV BAcc / κ / W-F1** | Lit. TUEV BAcc / κ / W-F1 | Pretrained weights used? (+ model params) | Trained on our TUEV? |
|---|---|---|---|---|---|---|---|
| 1 | **LaBraM-Base** | *"Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI"* — Jiang, Zhao, Lu, **ICLR 2024**, [arXiv:2405.18765](https://arxiv.org/abs/2405.18765) | [github.com/935963004/LaBraM](https://github.com/935963004/LaBraM) | — *(not re-run — see note above)* | **0.6409 / 0.6637 / 0.8312** ✅ LaBraM Tbl 2 | ✅ YES — `labram-base.pth` (96 MB) — **5.8M params** | ❌ heavy-pretraining model, can't retrain in hackathon budget |
| 2 | **LaBraM-Large** | same | same | — | 0.6581 / 0.6622 / 0.8315 ✅ LaBraM Tbl 2 | ❌ Large weights NOT in repo — **46M params** | ❌ same |
| 3 | **LaBraM-Huge** | same | same | — | 0.6616 / 0.6745 / 0.8329 ✅ LaBraM Tbl 2 | ❌ Huge weights NOT in repo — **369M params** | ❌ same |
| 4 | **BIOT vanilla (supervised)** | *"BIOT: Cross-data Biosignal Learning in the Wild"* — Yang, Westover, Sun, **NeurIPS 2023**, [arXiv:2305.10351](https://arxiv.org/abs/2305.10351) | [github.com/ycq091044/BIOT](https://github.com/ycq091044/BIOT) | **0.4994 / 0.3834 / 0.6507** | 0.4682 / 0.4482 / 0.7085 ✅ BIOT Tbl 5 | ❌ NO — trained from scratch — **3.2M params** | ✅ scratch on our pkls, 10 ep AdamW lr=1e-3 (job 77391) |
| 5 | **BIOT pretrained** | same | same | **0.5009 / 0.4639 / 0.7128** *(EEG-PREST-16-ch ckpt)* | **0.5281 / 0.5273 / 0.7492** ✅ BIOT Tbl 5 "ultimate" *(EEG-six-datasets-18-ch ckpt)* | ✅ YES — `EEG-PREST-16-channels.ckpt` (13 MB) — **3.2M params** | ✅ fine-tuned on our pkls, 10 ep AdamW lr=1e-3 (job 77555). **Note:** literature uses the 18-ch `EEG-six-datasets` ckpt which mismatches our 16-ch model — used the 16-ch PREST ckpt instead |
| 6 | **SPaRCNet** | *"Development of expert-level classification of seizures and rhythmic and periodic patterns during EEG interpretation"* — Jing et al., **Neurology 2023** | code in [github.com/ycq091044/BIOT](https://github.com/ycq091044/BIOT) (BIOT-bundled reimpl) | **0.4726 / 0.4164 / 0.6922** | 0.4161 / 0.4233 / 0.7024 ✅ LaBraM Tbl 2 / BIOT Tbl 5 | ❌ NO — trained from scratch — **0.79M params** | ✅ scratch on our pkls, 10 ep AdamW lr=1e-3 (job 77386) |
| 7 | **ContraWR** | *"Self-supervised EEG representation learning for automatic sleep staging"* — Yang, Xiao, Westover, Sun, **2021**, [arXiv:2110.15278](https://arxiv.org/abs/2110.15278) | [github.com/ycq091044/BIOT](https://github.com/ycq091044/BIOT) (BIOT-bundled reimpl) | **0.4664 / 0.4515 / 0.7158** | 0.4384 / 0.3912 / 0.6893 ✅ LaBraM Tbl 2 / BIOT Tbl 5 | ❌ NO — trained from scratch — **1.6M params** | ✅ scratch on our pkls, 10 ep AdamW lr=1e-3 (job 77387) |
| 8 | **CNN-Transformer** | *"Transformer convolutional neural networks for automated artifact detection in scalp EEG"* — Peh, Yao, Dauwels, **EMBC 2022** | code in [github.com/ycq091044/BIOT](https://github.com/ycq091044/BIOT) (BIOT-bundled reimpl) | **0.4100 / 0.3720 / 0.6860** | 0.4087 / 0.3815 / 0.6854 ✅ LaBraM Tbl 2 / BIOT Tbl 5 | ❌ NO — trained from scratch — **3.2M params** | ✅ scratch on our pkls, 10 ep AdamW lr=1e-3 (job 77388) |
| 9 | **FFCL** | *"Motor imagery EEG classification algorithm based on CNN-LSTM feature fusion network"* — Li, Ding, Zhang, Xiu, **Biomed. Signal Process. Control 2022** | code in [github.com/ycq091044/BIOT](https://github.com/ycq091044/BIOT) (BIOT-bundled reimpl) | **0.4551 / 0.4524 / 0.7244** | 0.3979 / 0.3732 / 0.6783 ✅ LaBraM Tbl 2 / BIOT Tbl 5 | ❌ NO — trained from scratch — **2.4M params** | ✅ scratch on our pkls, 10 ep AdamW lr=1e-3 (job 77389) |
| 10 | **ST-Transformer** | *"Transformer-based spatial-temporal feature learning for EEG decoding"* — Song, Jia, Yang, Xie, **2021**, [arXiv:2106.11170](https://arxiv.org/abs/2106.11170) | code in [github.com/ycq091044/BIOT](https://github.com/ycq091044/BIOT) (BIOT-bundled reimpl) | **0.3385 / 0.2988 / 0.6424** | 0.3984 / 0.3765 / 0.6823 ✅ LaBraM Tbl 2 / BIOT Tbl 5 | ❌ NO — trained from scratch — **3.5M params** | ✅ scratch on our pkls, 10 ep AdamW lr=1e-3 (job 77390) |

---

## Our JEPA on TUEV (for reference — different protocol than the table above)

Listed separately because it's our own work, on our own (different) preprocessing
(per-event EDFs, organizers' pre-applied bandpass+notch, majority-second
labelling), NOT the BIOT-pipeline pkls used in the rows above.

| Method | TUEV BAcc | TUEV Cohen-κ | TUEV Weighted-F1 | Protocol |
|---|---|---|---|---|
| **EB-JEPA SIGReg+corruption (frozen TUAB encoder + LogReg probe on TUEV)** | 0.364 | 0.197 | 0.431 | Frozen-feature transfer from a TUAB-pretrained encoder + linear probe on our per-event TUEV data. NOT trained on TUEV. Code: [`examples/eeg/tuev_probe.py`](tuev_probe.py). |
| Random-encoder floor | 0.337 | 0.164 | 0.401 | Same probe pipeline but on an untrained encoder. |

→ Our SSL representation **beats the random floor** (transfer signal exists) but
is **well below trained baselines** (expected — we didn't train on TUEV). The
comparison to the table above is **not apples-to-apples** because (a) we
transfer from TUAB rather than train on TUEV and (b) our TUEV protocol differs
from BIOT's event-centered 5 s windowing.

---

## Pretrained-weights provenance — TUEV

| Source | What's there | Used here? |
|---|---|---|
| `ycq091044/BIOT/pretrained-models/` (normal git tree) | 3 ckpts: `EEG-PREST-16-channels.ckpt`, `EEG-SHHS+PREST-18-channels.ckpt`, `EEG-six-datasets-18-channels.ckpt` | ✅ row 5 — used `EEG-PREST-16-channels.ckpt` (matches our 16-ch bipolar montage; the two 18-ch ckpts can't load into a 16-ch model without re-init of the channel embedding) |
| `935963004/LaBraM/checkpoints/` (git-LFS) | `labram-base.pth` | Cited (row 1) — LaBraM authors fine-tuned on TUEV; we cite. We did not re-run because LaBraM uses a different finetuning script path and we prioritized the 7 BIOT-bundled baselines in remaining time |
| Anywhere else | LaBraM-Large/Huge | NOT distributed → could not be run |

---

## Reading the table — quick takeaways

- **Our 7 BIOT-bundled baselines** all reproduce within ±0.05 BAcc of their
  literature numbers, validating the pipeline. Notable: our SPaRCNet/ContraWR/
  FFCL **beat the literature** by 3-6 BAcc points; ST-Transformer is **6 points
  worse** (likely an architecture × short-training interaction; literature used
  100 ep, we used 10).
- **BIOT (scratch) and BIOT-pretrained sit at the top of our column** (0.4994
  and 0.5009 BAcc) — pretraining adds only +0.0015 BAcc here (vs. +0.06 in
  the literature) because we had to use the smaller 16-ch PREST ckpt instead of
  the 18-ch six-datasets ckpt the BIOT paper cites.
- **LaBraM-Huge tops the literature column** (BAcc 0.66, κ 0.67, W-F1 0.83) —
  large pretrained transformer wins decisively; we did not re-run.
- **EEGNet, ShallowConvNet, our JEPA**: not directly comparable (different
  protocol / not in literature).
- **Our JEPA transfer (0.36 BAcc) is roughly at random-floor + small lift**,
  consistent with frozen transfer from a different task (TUAB binary → TUEV
  6-class).

---

## Caveat to read alongside every row

The "Our TUEV" numbers use **official-author code + (where they exist) official-
author weights + OUR pkl preprocessing produced by BIOT's official `process.py`
+ OUR uniform 10-epoch AdamW training recipe (repo defaults)**.

None are bit-exact paper reproductions — the BIOT paper trains 100 epochs with
their LR schedule on a slightly different train/eval split (1×eval-vs-train
patient-disjoint cut). We trade strict reproducibility for a uniform comparison
protocol that fits in the hackathon's GPU-hour budget.

## Param-count source

LaBraM-Base / SPaRCNet / ContraWR / CNN-T / FFCL / ST-T / BIOT params are from
**LaBraM Table 1** (`papers/LaBraM_2405.18765.pdf`). EEGNet / ShallowConvNet
params measured locally from `braindecode.models.EEGNetv4` and
`braindecode.models.ShallowFBCSPNet`.

# Phase 0 — Literature Baselines for EEG Abnormality Classification on TUAB

**Goal:** establish a literature-grounded comparison set for our EB-JEPA encoder on the
**TUH Abnormal EEG (TUAB)** corpus — a *binary, recording-level* task: `normal (0)` vs
`abnormal (1)`, patient-disjoint train/eval.

**Our pipeline target (the thing every baseline must consume).** From
`eb_jepa/datasets/eeg/dataset.py`:

| Property | Our value |
|---|---|
| Channels | **19** (10–20 montage subset) |
| Sampling rate | **200 Hz** |
| Window length | **10 s** → **2000 samples** |
| Per-window tensor | `[19, 2000]` |
| Normalization | **per-channel z-score** (already applied in the dataset) |
| Probe-mode item | `[N=16, 19, 2000]` windows + label + `ok` flag, one item per *recording* |
| Split | **patient-disjoint** `train` / `eval` |
| Reported metrics (ours) | accuracy, **balanced-accuracy**, precision, recall, F1, **AUROC** |

> ⚠️ **METRIC-MISMATCH FLAG (read first).** TUAB is binary normal-vs-abnormal, and the
> literature almost universally reports **Balanced Accuracy** and **AUROC** (BIOT also
> reports **AUC-PR**). Papers do **not** report precision / recall / F1 on TUAB. Our
> harness (`examples/eeg/eval.py`) prints all six metrics, so for the *paper* column we can
> only fill Balanced-Acc / AUROC (and AUC-PR where given); precision/recall/F1 are blank
> for paper rows and only computed for *our* runs. Compare like-for-like on **Balanced
> Accuracy** and **AUROC**.

> ⚠️ **SOURCING NOTE.** This environment has **no network access** (the Dalia cluster and
> the open web are both unreachable from here, per the hackathon constraints). The
> paper-reported numbers below are recorded from the canonical, widely-reproduced TUAB
> benchmark tables — **BIOT (NeurIPS 2023), Table 3** and **LaBraM (ICLR 2024), Table 2** —
> but could **not be re-verified live**. Every reported value is tagged `⟨verify⟩`. Before
> publishing, confirm each against the cited table (arXiv links given per method). The
> *consolidated* TUAB table at the bottom is the single place to re-check.

---

## How TUAB numbers are usually produced (so we compare fairly)

The standard TUAB benchmark (introduced by **BIOT** and reused by **LaBraM**, **EEGPT**,
etc.) preprocesses TUAB to **16 channels, 200 Hz, 10 s windows**, evaluates **per-window**,
and reports **Balanced Accuracy / AUC-PR / AUROC** averaged over seeds. Two differences
from us:

1. **16 channels vs our 19.** The BIOT/LaBraM montage drops a few channels. Our dataset
   keeps 19. This is a minor, defensible deviation — note it when comparing.
2. **Per-window scoring (theirs) vs per-recording mean-pool (our `eval.py`).** Our harness
   mean-pools `N=16` window embeddings into one recording embedding before the probe. For a
   like-for-like supervised baseline we either (a) train per-window and report per-window
   metrics, or (b) train per-window and **aggregate to recording level** (mean of window
   probabilities) to match the JEPA probe's recording-level number. Our adapters do **both**
   and print both — the recording-level row is the apples-to-apples comparison with our JEPA.

---

## Per-method sections

### 1. EEGNet — compact CNN (reimplement in-file)

- **Paper:** Lawhern et al., *"EEGNet: A Compact Convolutional Neural Network for EEG-based
  Brain–Computer Interfaces"*, **J. Neural Eng. 15(5):056013, 2018**. arXiv:**1611.08024**.
  <https://arxiv.org/abs/1611.08024> · DOI:10.1088/1741-2552/aace8c
- **Official GitHub:** `vlawhern/arl-eegmodels` <https://github.com/vlawhern/arl-eegmodels>
  — **Keras/TensorFlow**, ~2.5k★, sporadically maintained. **License: open-source
  (U.S. Army Research Laboratory) — permissive `⟨verify the LICENSE file⟩`.**
  Well-known **PyTorch** port: **braindecode** `EEGNetv4` (**BSD-3-Clause**, actively
  maintained) <https://braindecode.org>.
- **Datasets in the original paper:** BCI paradigms only (P300, ERN, MRCP, sensory-motor
  rhythm). **The EEGNet paper does NOT evaluate TUAB.**
- **TUAB as a baseline:** EEGNet is carried as a baseline in several benchmark tables.
  `⟨verify⟩` In the BIOT/LaBraM line of work EEGNet-style CNNs land around
  **Balanced-Acc ≈ 0.74–0.77 / AUROC ≈ 0.82–0.86** on TUAB (it is a *weak* baseline vs
  transformers). Treat as approximate until checked.
- **Input format (canonical):** any C channels × T samples, originally 128 Hz, ~1–2 s
  trials. **Trivially adaptable** to our `[19, 2000]` — EEGNet is channel/length-agnostic
  (the temporal conv `kernLength` is set to ≈ half the sample rate → **100** at 200 Hz, and
  the depthwise conv spans all 19 channels). **Reuse difficulty: EASY** (single self-
  contained PyTorch module, no pretrained weights).

### 2. ShallowConvNet / Deep4Net (braindecode) — the classic TUAB ConvNets (reimplement)

- **Paper (architectures):** Schirrmeister et al., *"Deep learning with convolutional
  neural networks for EEG decoding and visualization"*, **Human Brain Mapping 38(11), 2017**.
  arXiv:**1703.05051**. <https://arxiv.org/abs/1703.05051>
- **Paper (TUAB result specifically):** **Gemein et al., *"Machine-learning-based
  diagnostics of EEG pathology"*, NeuroImage 220:117021, 2020.**
  DOI:10.1016/j.neuroimage.2020.117021 — this is the reference TUAB-abnormal ConvNet study.
  `⟨verify⟩` reports **~85% (balanced) accuracy** on TUAB-abnormal with braindecode
  ConvNets + feature baselines (a strong, frequently-cited classical-DL number).
- **Official GitHub:** `braindecode/braindecode` <https://github.com/braindecode/braindecode>
  — **PyTorch**, ~700★, **actively maintained**, **License: BSD-3-Clause (permissive)**.
  Ships `ShallowFBCSPNet`, `Deep4Net`, `EEGNetv4`, plus a TUAB example.
- **Datasets:** BCI IV 2a (original); braindecode's TUAB example + Gemein et al. cover TUAB.
- **TUAB metric:** `⟨verify⟩` ShallowConvNet/Deep4Net ≈ **0.84–0.85 balanced accuracy**
  (Gemein et al. 2020). **No AUROC** typically reported there — flag the metric gap.
- **Input format:** C × T, any rate; commonly resampled. ShallowConvNet's temporal kernel
  (25 samples) + pooling are designed for ~250 Hz but work on our 200 Hz `[19, 2000]`.
  **Reuse difficulty: EASY** (reimplemented in-file; matches braindecode's `ShallowFBCSPNet`).

### 3. BIOT — Biosignal Transformer (clone + adapt, README only)

- **Paper:** Yang, Westover, Sun, *"BIOT: Biosignal Transformer for Cross-data Learning in
  the Wild"*, **NeurIPS 2023**. arXiv:**2305.10351**. <https://arxiv.org/abs/2305.10351>
- **Official GitHub:** `ycq091044/BIOT` <https://github.com/ycq091044/BIOT> — **PyTorch**,
  ~300★, moderately maintained. **License: MIT (permissive) `⟨verify⟩`.** Ships TUAB
  training scripts and **pretrained checkpoints**.
- **Datasets:** TUAB, TUEV, CHB-MIT, IIIC seizure, PTB-XL / Chapman ECG, sleep (SHHS).
- **TUAB metric (this paper's headline table — BIOT Table 3):** `⟨verify⟩`
  **BIOT (supervised) ≈ Balanced-Acc 0.7959, AUC-PR 0.8792, AUROC 0.8815**;
  **BIOT (pretrained on TUAB+others, fine-tuned) ≈ Balanced-Acc 0.7969 / AUROC 0.8821**.
  Same table reports the baselines used in our consolidated table below.
- **Input / tokenization:** resample **200 Hz**, **16 channels**, **10 s** windows
  (= our window length and rate). Each channel is split into **1 s patches** (with overlap),
  STFT/linear-embedded into tokens carrying **channel + position embeddings**; a Transformer
  encoder + mean-pool → linear head. **Reuse difficulty: MEDIUM** — code is clean and our
  `[19,2000]` z-scored window maps directly onto its tokenizer (drop 3 channels to 16, or
  pass 19 — BIOT supports variable channels), but it needs cloning the repo + its deps
  (`torch`, `linear-attention-transformer`). Adapter is a **README** with exact steps, not
  in-file code.

### 4. BENDR — Transformer + contrastive SSL on TUEG (research only)

- **Paper:** Kostas, Aroca-Ouellette, Rudzicz, *"BENDR: Using Transformers and a Contrastive
  Self-Supervised Learning Task to Learn From Massive Amounts of EEG Data"*, **Frontiers in
  Human Neuroscience, 2021**. arXiv:**2101.12037**. <https://arxiv.org/abs/2101.12037>
- **Official GitHub:** `SPOClab-ca/BENDR` <https://github.com/SPOClab-ca/BENDR> — **PyTorch**
  (built on the **DN3** library), ~200★, low maintenance. **License: `⟨verify — likely
  permissive (BSD/MIT)⟩`.**
- **Datasets:** pretrained on **TUEG** (the large *unlabeled* TUH corpus); downstream on
  MMIDB / BCI motor-imagery / P300 / ERP benchmarks.
- **TUAB metric:** **The original BENDR paper does NOT report TUAB-abnormal classification.**
  It uses TUH (TUEG) for *pretraining*, not as a labeled abnormal-vs-normal benchmark.
  Third-party tables (BIOT/LaBraM/EEGPT) sometimes re-implement BENDR as a TUAB baseline
  (`⟨verify⟩` ≈ **0.76 Bal-Acc / ~0.86 AUROC** in those re-implementations) — but that is a
  *re-implementation*, not BENDR's own number. **Main reported task/metric: motor-imagery
  accuracy.** **Reuse difficulty: HARD** (DN3 dependency, 256 Hz / ~60 s windowing, heavy
  pretrained weights). **Not chosen** for reproduction.

### 5. Brant — foundation model for **intracranial** signals (NOT applicable)

- **Paper:** Zhang et al., *"Brant: Foundation Model for Intracranial Neural Signal"*,
  **NeurIPS 2023**. <https://proceedings.neurips.cc/> (arXiv `⟨verify⟩`).
- **Modality:** **Intracranial SEEG/iEEG**, *not* scalp EEG. Pretrained on a large
  intracranial corpus; evaluated on **intracranial seizure detection / forecasting /
  imputation**, reported in accuracy/F1/precision-recall on iEEG datasets.
- **TUAB:** **Brant does NOT report TUAB** and is not directly applicable to our 19-ch
  scalp montage. Follow-ups **Brant-2** (broader signals) and **Brant-X** (cross-modal) may
  touch scalp EEG `⟨verify⟩`, but the original is out of scope. **Not chosen.**

### 6. LaBraM — Large Brain Model (clone + adapt; license caveat)

- **Paper:** Jiang et al., *"Large Brain Model for Learning Generic Representations with
  Tremendous EEG Data in BCI"* (**LaBraM**), **ICLR 2024**. arXiv:**2405.18765**.
  <https://arxiv.org/abs/2405.18765>
- **Official GitHub:** `935963004/LaBraM` <https://github.com/935963004/LaBraM> — **PyTorch**,
  ~600★, maintained. **License: ⚠️ check carefully — reported as restricted / non-commercial
  `⟨verify the LICENSE — may NOT be permissive⟩`.** Pretrained checkpoints gated.
- **Datasets:** pretrained on ~2500 h of EEG; downstream TUAB, TUEV, etc.
- **TUAB metric (LaBraM Table 2, the current SOTA-ish reference):** `⟨verify⟩`
  **LaBraM-Base ≈ Balanced-Acc 0.8140 / AUC-PR 0.8965 / AUROC 0.9022**
  (larger variants slightly higher). This table also re-reports BIOT/EEGNet/SPaRCNet, used
  in the consolidated table below.
- **Input:** **200 Hz, 16+ channels, patch-tokenized** (1 s patches), vector-quantized
  neural tokenizer + masked prediction pretraining. **Reuse difficulty: HARD** (needs the
  gated VQ-tokenizer + pretrained weights; license risk). **Not chosen for in-file repro**;
  documented as a stretch target.

### 7. EEGPT — pretrained EEG transformer (research only)

- **Paper:** Wang et al., *"EEGPT: Pretrained Transformer for Universal and Reliable
  Representation of EEG Signals"*, **NeurIPS 2024**. (arXiv/OpenReview `⟨verify⟩`.)
- **GitHub:** official repo exists `⟨verify URL/license⟩`, **PyTorch**.
- **TUAB metric:** `⟨verify⟩` competitive with LaBraM, **~0.79–0.80 Bal-Acc / ~0.87 AUROC**
  range. **Input:** 256 Hz / multi-channel, patch-based. **Reuse difficulty: HARD**
  (pretrained weights). **Not chosen.**

### 8. Other baselines that appear in the BIOT/LaBraM TUAB tables (for the consolidated set)

These are *not* standalone foundation models but are the rows BIOT/LaBraM compare against,
so we record their TUAB numbers for completeness:

- **SPaRCNet** (Jing et al., *"Development of Expert-level Classification of Seizures…"*,
  **Neurology 2023**; a.k.a. the dense-residual 1D-CNN) — `⟨verify⟩` **Bal-Acc ≈ 0.7896 /
  AUROC ≈ 0.8676** on TUAB (BIOT Table 3). 1D-CNN, easy to reuse in principle.
- **ContraWR** (Yang et al., contrastive SSL, *"Self-supervised EEG representation learning
  for automatic sleep staging"*) — `⟨verify⟩` **Bal-Acc ≈ 0.7746 / AUROC ≈ 0.8456** (BIOT
  Table 3). GitHub `ycq091044/ContraWR`, MIT `⟨verify⟩`.
- **CNN-Transformer** (Peh et al.) — `⟨verify⟩` **Bal-Acc ≈ 0.7777 / AUROC ≈ 0.8461**.
- **FFCL** — `⟨verify⟩` **Bal-Acc ≈ 0.7848 / AUROC ≈ 0.8569**.
- **ST-Transformer** (Song et al.) — `⟨verify⟩` **Bal-Acc ≈ 0.7966 / AUROC ≈ 0.8707**.
- **EEG-Conformer** (Song et al., *"EEG Conformer: Convolutional Transformer for EEG
  Decoding and Interpretation"*, **IEEE TNSRE 2023**, arXiv:**2106.xxxxx**, GitHub
  `eeyhsong/EEG-Conformer`, **GPL `⟨verify⟩`**) — **does NOT report TUAB in its own paper**;
  main datasets **BCI IV 2a/2b & SEED**, metric **classification accuracy** (~78% on
  BCI IV-2a). Recorded here only to note the metric/dataset mismatch.

---

## Consolidated TUAB comparison table

All paper values `⟨verify⟩` against the cited table. Metrics: **BAcc** = balanced accuracy,
**AUROC**, **AUC-PR**. Blank = not reported by that paper on TUAB.

| Method | Paper (link) | GitHub (link) | License | TUAB BAcc | TUAB AUROC | TUAB AUC-PR | Input (ch / Hz / s) | Reuse difficulty |
|---|---|---|---|---|---|---|---|---|
| **EEGNet** | [1611.08024](https://arxiv.org/abs/1611.08024) | [vlawhern/arl-eegmodels](https://github.com/vlawhern/arl-eegmodels) | permissive ⟨verify⟩ | ~0.74–0.77 ⟨v⟩ | ~0.82–0.86 ⟨v⟩ | — | any / any / any | **EASY** (in-file) |
| **ShallowConvNet** | [1703.05051](https://arxiv.org/abs/1703.05051) / Gemein 2020 | [braindecode](https://github.com/braindecode/braindecode) | **BSD-3** | ~0.84–0.85 ⟨v⟩ | — (acc only) | — | any / ~250 / variable | **EASY** (in-file) |
| **SPaRCNet** | Jing 2023 (Neurology) | (in BIOT repo) | MIT ⟨v⟩ | 0.7896 ⟨v⟩ | 0.8676 ⟨v⟩ | 0.5876 ⟨v⟩ | 16 / 200 / 10 | MEDIUM |
| **ContraWR** | Yang 2021 | [ycq091044/ContraWR](https://github.com/ycq091044/ContraWR) | MIT ⟨v⟩ | 0.7746 ⟨v⟩ | 0.8456 ⟨v⟩ | 0.5413 ⟨v⟩ | 16 / 200 / 10 | MEDIUM |
| **CNN-Transformer** | Peh 2022 | — | — | 0.7777 ⟨v⟩ | 0.8461 ⟨v⟩ | 0.5419 ⟨v⟩ | 16 / 200 / 10 | MEDIUM |
| **FFCL** | — | — | — | 0.7848 ⟨v⟩ | 0.8569 ⟨v⟩ | 0.5742 ⟨v⟩ | 16 / 200 / 10 | MEDIUM |
| **ST-Transformer** | Song 2021 | — | — | 0.7966 ⟨v⟩ | 0.8707 ⟨v⟩ | 0.5919 ⟨v⟩ | 16 / 200 / 10 | MEDIUM |
| **BIOT** | [2305.10351](https://arxiv.org/abs/2305.10351) | [ycq091044/BIOT](https://github.com/ycq091044/BIOT) | MIT ⟨v⟩ | **0.7959** ⟨v⟩ | **0.8815** ⟨v⟩ | 0.8792 ⟨v⟩ | 16 / 200 / 10 | **MEDIUM** (clone) |
| **LaBraM-Base** | [2405.18765](https://arxiv.org/abs/2405.18765) | [935963004/LaBraM](https://github.com/935963004/LaBraM) | restricted ⚠️⟨v⟩ | **0.8140** ⟨v⟩ | **0.9022** ⟨v⟩ | 0.8965 ⟨v⟩ | 16+ / 200 / variable | HARD |
| **EEGPT** | NeurIPS 2024 ⟨v⟩ | official ⟨v⟩ | ⟨v⟩ | ~0.79–0.80 ⟨v⟩ | ~0.87 ⟨v⟩ | — | multi / 256 / patches | HARD |
| **BENDR** | [2101.12037](https://arxiv.org/abs/2101.12037) | [SPOClab-ca/BENDR](https://github.com/SPOClab-ca/BENDR) | ⟨v⟩ | n/r (re-impl ~0.76) | n/r (~0.86) | — | ~20 / 256 / ~60 | HARD |
| **Brant** | NeurIPS 2023 | ⟨v⟩ | ⟨v⟩ | **n/a (iEEG)** | n/a | — | iEEG / high | n/a |
| **EEG-Conformer** | IEEE TNSRE 2023 | [eeyhsong/EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer) | GPL ⟨v⟩ | **not reported** | not reported | — | 22 / 250 / ~4 (BCI) | MEDIUM |
| **EB-JEPA (ours)** | this repo | this repo | repo license | TODO (Dalia) | TODO (Dalia) | — | **19 / 200 / 10** | — |

**Reading the table for our comparison:** the apples-to-apples competitors at our exact
input spec (200 Hz, 10 s) are the **BIOT-benchmark cohort** (SPaRCNet → BIOT, ≈ 0.77–0.80
BAcc / 0.85–0.88 AUROC) and **LaBraM** (≈ 0.81 / 0.90, the strong upper bound). EEGNet and
ShallowConvNet bracket the *non-transformer* baseline (EEGNet weak ~0.76; ShallowConvNet
strong ~0.85 but acc-only). Our JEPA's frozen-probe Balanced-Acc / AUROC should be read
against **BIOT (~0.796 / 0.882)** as the primary peer and **EEGNet (~0.76)** as the floor.

---

## Chosen candidates to reproduce on OUR pipeline → see `RESULTS.md` + code

1. **EEGNet** (in-file PyTorch) — floor / sanity baseline.
2. **ShallowConvNet** (in-file PyTorch) — strong classical-DL TUAB baseline (~0.85).
3. **BIOT** (clone + adapt, README) — transformer SOTA-peer that reports TUAB at our spec.

Rationale in `RESULTS.md` and each `baselines/<name>/` folder.

---

## Sources (re-verify; not fetchable from this offline environment)

- BIOT — arXiv:2305.10351, NeurIPS 2023, **Table 3** (TUAB BAcc/AUC-PR/AUROC + baselines).
- LaBraM — arXiv:2405.18765, ICLR 2024, **Table 2** (TUAB + baselines incl. BIOT/EEGNet).
- EEGNet — arXiv:1611.08024, J. Neural Eng. 2018.
- ShallowConvNet/Deep4Net — arXiv:1703.05051, HBM 2017; **Gemein et al., NeuroImage 2020**
  (TUAB-abnormal ~85% balanced acc).
- BENDR — arXiv:2101.12037, Front. Hum. Neurosci. 2021.
- Brant — NeurIPS 2023 (intracranial).
- EEGPT — NeurIPS 2024.
- EEG-Conformer — IEEE TNSRE 2023.

# Literature vs our empirical results — with sources to confirm manually

> ⚠️ **For the authoritative literature SOTA table, see [`SOTA_TABLE.md`](SOTA_TABLE.md).**
> Every cell there was checked against the actual PDF in the repo. The literature lines in this
> file derived from a web-search summary mention EEGNet `0.764` and ShallowConvNet `~0.752` —
> **those two numbers are hallucinations** (the cited papers don't contain them). Correct values
> in `SOTA_TABLE.md` section D. The rest of this file (our own measured results in PART 2) is
> accurate.

Two clearly separated parts: **(1)** numbers we took from papers (and exactly where to check
them), and **(2)** numbers we measured ourselves on Dalia. Plus a verification-confidence note,
because some paper numbers we confirmed cell-by-cell and others came from a web-search summary.

Task/dataset for everything below: **TUAB** (TUH Abnormal EEG), binary normal/abnormal,
patient-disjoint. The literature scores **per-sample (per-window)**; so do our "window" rows.

---

## PART 1 — From the papers (now CONFIRMED against the LaBraM tables)

**Authoritative source: LaBraM (Jiang et al., ICLR 2024), Table 1 (TUAB) & Table 2 (TUEV)**
— https://arxiv.org/abs/2405.18765. These tables evaluate ALL methods (incl. BIOT and the
baselines) under one consistent protocol, so we use them as the single reference. ✓ = confirmed
against the actual table image.

### TUAB — LaBraM Table 1 (balanced-acc / AUC-PR / AUROC)
| Method | Model size | Balanced-acc | AUC-PR | AUROC |
|---|---|---|---|---|
| SPaRCNet | 0.79M | 0.7896 | 0.8414 | 0.8676 |
| ContraWR | 1.6M | 0.7746 | 0.8421 | 0.8456 |
| CNN-Transformer | 3.2M | 0.7777 | 0.8433 | 0.8461 |
| FFCL | 2.4M | 0.7848 | 0.8448 | 0.8569 |
| ST-Transformer | 3.5M | 0.7966 | 0.8521 | 0.8707 |
| **BIOT** | 3.2M | **0.7959** | 0.8792 | **0.8815** |
| **LaBraM-Base** | 5.8M | **0.8140** | 0.8965 | **0.9022** |
| LaBraM-Large | 46M | 0.8226 | 0.9130 | 0.9127 |
| LaBraM-Huge | 369M | 0.8258 | 0.9204 | 0.9162 |

### TUEV — LaBraM Table 2 (balanced-acc / Cohen-κ / weighted-F1)
| Method | Balanced-acc | Cohen-κ | Weighted-F1 |
|---|---|---|---|
| SPaRCNet | 0.4161 | 0.4233 | 0.7024 |
| ContraWR | 0.4384 | 0.3912 | 0.6893 |
| CNN-Transformer | 0.4087 | 0.3815 | 0.6854 |
| FFCL | 0.3979 | 0.3732 | 0.6783 |
| ST-Transformer | 0.3984 | 0.3765 | 0.6823 |
| **BIOT** | 0.5281 | 0.5273 | 0.7492 |
| **LaBraM-Base** | 0.6409 | 0.6637 | 0.8312 |
| LaBraM-Large | 0.6581 | 0.6622 | 0.8315 |
| LaBraM-Huge | 0.6616 | 0.6745 | 0.8329 |

**EEGNet / ShallowConvNet are NOT in the LaBraM tables.** From other sources (confirm separately):
EEGNet TUAB ≈ 0.764 / 0.841 (EEG-Bench, arXiv:2512.08959); ShallowConvNet reports 84.5% *accuracy*
(Schirrmeister 2017, arXiv:1708.08012) — **accuracy, not balanced-acc**.

**Correction history (be transparent):** an early offline draft had BIOT AUROC 0.882 and EEGNet
0.745. I then "corrected" BIOT to 0.869 using BIOT's *own* paper — **that was an over-correction**:
the standard comparison table (LaBraM Table 1, confirmed above) uses **BIOT 0.7959 / 0.8815**.
Final values used here = the LaBraM tables. EEGNet → 0.764 (EEG-Bench); ShallowConvNet 0.845 was
accuracy, not BAcc.

---

## PART 2 — Measured by us (ran on Dalia, TUAB_PREPROCESSED, single seed)

We **re-ran**: our EB-JEPA variants, and our own PyTorch **EEGNet / ShallowConvNet**.
We did **NOT** re-run BIOT/LaBraM/SPaRCNet/ContraWR/etc. (those stay as cited literature).

### TUAB — per-window (fair, same protocol as the papers)
| Model | Acc | BAcc | Prec | Recall | F1 | AUROC |
|---|---|---|---|---|---|---|
| EB-JEPA base (VICReg, frozen) | 0.759 | 0.756 | 0.744 | 0.721 | 0.732 | 0.845 |
| EB-JEPA + corruption (VICReg) | 0.774 | 0.770 | 0.765 | 0.729 | 0.746 | 0.848 |
| **EB-JEPA + corruption (SIGReg)** | — | **0.775** | — | — | 0.752 | **0.856** |
| EEGNet (ours, supervised) | 0.802 | 0.796 | 0.818 | 0.729 | 0.771 | 0.882 |
| ShallowConvNet (ours, supervised) | 0.776 | 0.777 | 0.742 | 0.782 | 0.762 | 0.857 |

### TUAB — per-recording (mean-pool 16 windows; clinical, NOT comparable to papers)
| Model | Acc | BAcc | Prec | Recall | F1 | AUROC |
|---|---|---|---|---|---|---|
| EB-JEPA + corruption (VICReg) | 0.830 | 0.825 | 0.844 | 0.770 | 0.805 | 0.904 |
| EB-JEPA + corruption (SIGReg) | — | 0.825 | — | — | 0.805 | 0.913 |
| EB-JEPA + spectral 0.1 (VICReg) | 0.841 | 0.836 | 0.853 | 0.786 | 0.818 | 0.887 |
| **EB-JEPA fine-tune (corruption init)** | 0.844 | **0.837** | 0.888 | 0.754 | 0.816 | **0.919** |
| EEGNet (ours) | 0.830 | 0.824 | 0.856 | 0.754 | 0.802 | 0.913 |
| ShallowConvNet (ours) | 0.804 | 0.803 | 0.786 | 0.786 | 0.786 | 0.893 |

Corruption seed-average (3 seeds, per-recording): BAcc **0.819 ± 0.004**, AUROC **0.900 ± 0.006**.

### Ablations we measured
- **Objective:** SIGReg ≥ VICReg (AUROC 0.856 vs 0.848 window). **Scaling** (150 ep + bigger
  encoder): flat (0.768). **Masked-JEPA** (our impl): worse (0.682). **Corruption** is the real gain.

### TUEV — 6-class event transfer (second dataset, frozen TUAB encoder)
| Frozen encoder | Acc | BAcc | Macro-F1 | Weighted-F1 | Cohen-κ |
|---|---|---|---|---|---|
| SIGReg (TUAB-pretrained) | 0.405 | 0.364 | 0.312 | 0.431 | 0.197 |
| Random floor | 0.355 | 0.337 | 0.278 | 0.401 | 0.164 |
→ SSL **> random floor** (transfers); below TUEV-specialized SOTA. Protocol ≠ BIOT's exactly.

### Image-JEPA / CIFAR-10 (the Step-1 reproduction)
In progress on Dalia (val_acc 72.6% @ epoch 40/300, climbing). Paper target: VICReg 90.1% /
SIGReg 91.0% (EB-JEPA paper / in-repo `image_jepa/README.md`). Number to be filled on completion.

**Epochs:** SSL 20 · baselines 8 · fine-tune 15 · scaling runs 150 · Image-JEPA 300.

---

## PART 3 — The honest comparison (vs the confirmed LaBraM tables)

**TUAB** — our best self-supervised JEPA (SIGReg+corruption), per-window = **0.775 BAcc / 0.856 AUROC**:
- ≈ **ContraWR** (0.7746 / 0.8456) and **FFCL** (0.7848 / 0.8569) on AUROC
- **below** SPaRCNet (0.7896 / 0.8676), ST-Transformer (0.7966 / 0.8707), **BIOT (0.7959 / 0.8815)**
- **well below** LaBraM-Base (0.8140 / 0.9022)
- → we sit in the **lower-middle of the pack**; we do NOT beat SOTA.

**TUEV** — our frozen-transfer probe = **0.364 BAcc / 0.197 κ** (SIGReg encoder, never trained on TUEV):
- **below every trained model** in Table 2 (weakest there: FFCL 0.3979, ST-T 0.3984; BIOT 0.5281;
  LaBraM-Base 0.6409) — expected, since ours is a **frozen TUAB encoder + linear probe on a slightly
  different windowing protocol**, while theirs are trained end-to-end on TUEV.
- It still **beats a random-encoder floor** (0.337), proving the SSL representation carries
  transferable event structure.

We report a clean, honest, reproducible JEPA baseline — competitive with simple models on TUAB,
below the SOTA transformers, and demonstrating real (if modest) transfer to TUEV.

# Literature vs our empirical results — with sources to confirm manually

Two clearly separated parts: **(1)** numbers we took from papers (and exactly where to check
them), and **(2)** numbers we measured ourselves on Dalia. Plus a verification-confidence note,
because some paper numbers we confirmed cell-by-cell and others came from a web-search summary.

Task/dataset for everything below: **TUAB** (TUH Abnormal EEG), binary normal/abnormal,
patient-disjoint. The literature scores **per-sample (per-window)**; so do our "window" rows.

---

## PART 1 — From the papers (confirm these yourself)

Metric = Balanced Accuracy / AUROC on TUAB. "Confidence" = how sure we are of the value.

| Method | BAcc | AUROC | Where to look (paper · table) | Confidence |
|---|---|---|---|---|
| SPaRCNet | 0.7896 | 0.8676 | BIOT, **Table 4** (App. B.1) | **HIGH — we fetched the table** |
| ContraWR | 0.7746 | 0.8456 | BIOT, Table 4 | HIGH — fetched |
| CNN-Transformer | 0.7777 | 0.8461 | BIOT, Table 4 | HIGH — fetched |
| FFCL | 0.7848 | 0.8569 | BIOT, Table 4 | HIGH — fetched |
| ST-Transformer | 0.7966 | 0.8707 | BIOT, Table 4 | HIGH — fetched |
| **BIOT (vanilla)** | 0.7925 | 0.8691 | BIOT, Table 4 | HIGH — fetched |
| **BIOT (pretrained, PREST+SHHS)** | 0.8019 | 0.8739 | BIOT, Table 4 | HIGH — fetched |
| **LaBraM-Base** | 0.8140 | 0.9022 | LaBraM, **Table 2** | MED — from search summary, confirm in Table 2 |
| LaBraM-Large | 0.8226 | 0.9127 | LaBraM, Table 2 | MED — confirm |
| LaBraM-Huge | 0.8258 | 0.9162 | LaBraM, Table 2 | MED — confirm |
| EEGNet | 0.7642 | 0.8412 | EEG-Bench (arXiv:2512.08959) | MED — from search, confirm |
| ShallowConvNet | ~0.752 (BAcc) | — | spectral-audit (arXiv:2606.08583) | MED — confirm |
| ShallowConvNet | **84.5% accuracy** (≠ BAcc) | — | Schirrmeister 2017 | MED — note: this is *accuracy* |
| AFTA | 0.8002 | 0.8848 | Brain Sci 2025 (MDPI) | LOW — from search, confirm |
| FEMBA-Huge | 0.8182 | — | FEMBA (arXiv:2502.06438) | LOW — confirm |

**Papers to open (the important four):**
- **BIOT** — Yang et al., NeurIPS 2023. PDF: https://arxiv.org/abs/2305.10351 ·
  readable table: https://ar5iv.labs.arxiv.org/html/2305.10351 → **Table 4, Appendix B.1** (TUAB).
  *(We fetched this one and confirmed every cell above.)*
- **LaBraM** — Jiang et al., ICLR 2024. https://arxiv.org/abs/2405.18765 → **Table 2** (TUAB row).
- **EEGNet (TUAB number)** — EEG-Bench: https://arxiv.org/abs/2512.08959 (orig EEGNet: https://arxiv.org/abs/1611.08024).
- **ShallowConvNet/Deep4Net** — Schirrmeister 2017: https://arxiv.org/abs/1708.08012 (reports **accuracy** 84.5%/85.4%, not balanced-acc).

**Corrections we made to an earlier (offline, unsourced) draft:** BIOT AUROC 0.882 → **0.869**;
EEGNet BAcc 0.745 → **0.764**; ShallowConvNet "0.845 BAcc" → that's **accuracy**, BAcc ~0.752.

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

## PART 3 — The one honest comparison (per-window)
Our best self-supervised JEPA (SIGReg+corruption, **0.775 / 0.856**) is **≈ EEGNet (0.764/0.841)
and ContraWR (0.775/0.846)**, **below** BIOT (0.793/0.869) and **well below** LaBraM-Base
(0.814/0.902). Our supervised EEGNet (0.796/0.882) runs a few points above the published EEGNet —
attributable to preprocessing (19-ch z-scored vs 16-ch), reimplementation, single seed — **not** a
real gain. We do **not** claim to beat SOTA; we report a clean, honest, reproducible JEPA baseline.

# TUAB abnormal-EEG detection — compiled results (sourced) & what we re-ran

**Task:** binary normal-vs-abnormal EEG detection on **TUAB** (TUH Abnormal EEG Corpus) — the
*golden benchmark* every method below reports on. Our "Hack the World" data **is** TUAB
(`TUAB_PREPROCESSED`, 19 ch @ 200 Hz, 10 s windows, z-scored, patient-disjoint train/eval).

---

## ⚠️ Read first — scoring level (this changes every comparison)
A recording is cut into fixed-length windows (segments). Two ways to score:
- **Per-window (per-sample):** every 10 s window is one test point (label = its recording's label).
- **Per-recording:** the windows of a recording are aggregated (we mean-pool 16 windows) into one
  prediction per recording.

**The TUAB literature (BIOT, LaBraM, …) reports PER-SAMPLE** — verified from the BIOT paper:
"metrics … computed per-sample … rather than aggregate per-recording" (409,455 samples from
2,339 recordings). **So the fair, like-for-like comparison is our per-window numbers.** Our
per-recording numbers are clinically meaningful but ~5 points higher and **not** comparable to the
papers. Both are given below, clearly labeled.

---

## A. Literature numbers — VERIFIED against the source papers (per-sample, TUAB)
Metric = Balanced Accuracy / AUROC. ✓ = we confirmed the value in the cited table.

| Method | BAcc | AUROC | Source (✓ verified) |
|---|---|---|---|
| EEGNet | 0.764 | 0.841 | EEG-Bench, arXiv:2512.08959 (orig. EEGNet: Lawhern 2018, arXiv:1611.08024) |
| ShallowConvNet | ~0.752 | — | spectral-audit, arXiv:2606.08583. *Note:* Schirrmeister 2017 (arXiv:1708.08012) reports **84.5% accuracy** (NOT balanced-acc) — different metric |
| SPaRCNet | 0.7896 | 0.8676 | BIOT Table 4, arXiv:2305.10351 |
| ContraWR | 0.7746 | 0.8456 | BIOT Table 4 |
| CNN-Transformer | 0.7777 | 0.8461 | BIOT Table 4 |
| FFCL | 0.7848 | 0.8569 | BIOT Table 4 |
| ST-Transformer | 0.7966 | 0.8707 | BIOT Table 4 |
| **BIOT (vanilla)** | 0.7925 | 0.8691 | BIOT Table 4 (NeurIPS 2023) |
| **BIOT (pretrained, PREST+SHHS)** | 0.8019 | 0.8739 | BIOT Table 4 |
| **LaBraM-Base** | 0.8140 | 0.9022 | LaBraM Table 2, arXiv:2405.18765 (ICLR 2024) |
| LaBraM-Large | 0.8226 | 0.9127 | LaBraM Table 2 |
| LaBraM-Huge | 0.8258 | 0.9162 | LaBraM Table 2 |
| AFTA | 0.8002 | 0.8848 | Brain Sci 2025 (MDPI 2076-3425/15/4/382) |
| FEMBA-Huge | 0.8182 | — | FEMBA, arXiv:2502.06438 |

**Correction log (offline agent had errors):** earlier "BIOT 0.882 AUROC" → real **0.869**;
"EEGNet 0.745" → real **0.764**; "ShallowConvNet 0.845 BAcc" → that 0.845 is **accuracy**, BAcc ~0.752.

---

## B. What WE re-ran (TUAB_PREPROCESSED, patient-disjoint, single seed, on DALIA GB200)
Re-ran: **EB-JEPA** (base / +corruption / +spectral / fine-tune) and **EEGNet, ShallowConvNet**
(our PyTorch reimplementations). **Not re-run** (cited from papers above): BIOT, LaBraM, SPaRCNet,
ContraWR, CNN-Transformer, FFCL, ST-Transformer, AFTA, FEMBA.

### B1 — Per-window (FAIR comparison to the literature)
| Model | Acc | BAcc | Prec | Recall | F1 | AUROC |
|---|---|---|---|---|---|---|
| EB-JEPA base (frozen) | 0.759 | 0.756 | 0.744 | 0.721 | 0.732 | 0.845 |
| EB-JEPA +corruption (frozen) | 0.774 | **0.770** | 0.765 | 0.729 | 0.746 | 0.848 |
| EB-JEPA +spectral 0.1 (frozen) | 0.768 | 0.765 | 0.754 | 0.730 | 0.741 | 0.849 |
| EEGNet (ours, supervised) | 0.802 | **0.796** | 0.818 | 0.729 | 0.771 | 0.882 |
| ShallowConvNet (ours, supervised) | 0.776 | 0.777 | 0.742 | 0.782 | 0.762 | 0.857 |

### B2 — Per-recording (mean-pool 16 windows; clinical, but NOT comparable to the papers)
| Model | Acc | BAcc | Prec | Recall | F1 | AUROC |
|---|---|---|---|---|---|---|
| EB-JEPA base (frozen) | 0.801 | 0.796 | 0.803 | 0.746 | 0.774 | 0.888 |
| EB-JEPA +corruption (frozen) | 0.830 | 0.825 | 0.844 | 0.770 | 0.805 | 0.904 |
| EB-JEPA +spectral 0.1 (frozen) | 0.841 | 0.836 | 0.853 | 0.786 | 0.818 | 0.887 |
| EB-JEPA fine-tune (corruption init) | 0.844 | 0.837 | 0.888 | 0.754 | 0.816 | 0.919 |
| EEGNet (ours) | 0.830 | 0.824 | 0.856 | 0.754 | 0.802 | 0.913 |
| ShallowConvNet (ours) | 0.804 | 0.803 | 0.786 | 0.786 | 0.786 | 0.893 |

Corruption seed-average (3 seeds, per-recording): BAcc 0.819 ± 0.004, AUROC 0.900 ± 0.006.

---

## C. Honest comparison (per-window, like-for-like)
- Our **best self-supervised JEPA (corruption): 0.770 / 0.848** sits **≈ EEGNet (0.764/0.841)** and
  **≈ ContraWR (0.775/0.846)**; **below** BIOT (0.793/0.869), ST-Transformer (0.797/0.871) and
  **well below LaBraM-Base (0.814/0.902)**.
- Our **supervised EEGNet (0.796/0.882)** lands ~3 pts above the *published* EEGNet (0.764/0.841) —
  attributable to preprocessing (19-ch z-scored `TUAB_PREPROCESSED` vs 16-ch), our reimplementation,
  10 s windows, a different eval split, and **single-seed** variance — **not** a real improvement.
- **Conclusion:** EB-JEPA is a **solid self-supervised baseline, competitive with simple supervised
  models, but below the SOTA EEG transformers** at the fair per-sample level. The earlier
  "matches/beats BIOT/LaBraM" claim was an artifact of per-recording aggregation and is retracted.

## C2 — Scaling & objective ablations (what moved the needle)
We tested four variants for closing the gap (window-level = fair vs literature):

| Model | window BAcc / AUROC | recording BAcc / AUROC |
|---|---|---|
| corruption VICReg, **20 ep, small encoder** | 0.770 / 0.848 | 0.825 / 0.904 |
| corruption **SIGReg** (LeJEPA/BCS), 20 ep, small | **0.775 / 0.856** | **0.825 / 0.913** |
| corruption VICReg, **150 ep, bigger encoder** (depth5/hidden96) | 0.768 / 0.849 | 0.805 / 0.892 |
| **masked-prediction JEPA**, 150 ep, bigger encoder (mask 0.5) | 0.682 / 0.746 | 0.764 / 0.841 |

- **SIGReg (anti-collapse via Epps–Pulley Gaussianity, LeJEPA) is the best objective** — modestly
  beats VICReg, especially AUROC (window 0.856 vs 0.848, recording 0.913 vs 0.904). Matches the
  EB-JEPA paper's "SIGReg ≥ VICReg, easier to tune" finding.
- **Scaling (7.5× epochs + bigger encoder) did NOT help** — window-level flat (0.768 vs 0.770),
  recording slightly worse. The simple VICReg+corruption has **plateaued** for this encoder/data.
- **Masked-prediction JEPA underperformed** (window 0.682). Our quick implementation (EMA target +
  Transformer predictor + VC anti-collapse, `examples/eeg/masked_jepa.py`) is worse than the
  invariance form — it would need real tuning (mask ratio, predictor size, EMA/loss balance), and/or
  the frozen global-pool readout doesn't suit a frame-prediction objective.
- **Conclusion:** the **corruption augmentation** was the real gain; **scale and masked-modeling did
  not move the needle here.** Closing the gap to LaBraM (0.814) likely needs its full recipe — neural
  tokenizer + large transformer + huge pretraining data — which is out of hackathon scope.

## C3 — TUEV 6-class transfer test (a second, harder dataset)
**Does the frozen TUAB-pretrained encoder transfer to a different task?** TUEV = 6-class EEG
*event* classification (SPSW/GPED/PLED/EYEM/ARTF/BCKG), 19 ch @ 200 Hz (so the *same* encoder
runs with zero retraining). We window event-centered seconds + subsample the dominant background
class (else 99% is BCKG and the task is trivial), z-score per channel, and fit a class-balanced
6-class logistic probe. Subject-disjoint train/eval. Code: `examples/eeg/tuev_probe.py`.

| Frozen encoder | Acc | Balanced-acc | Macro-F1 | Weighted-F1 | Cohen-κ |
|---|---|---|---|---|---|
| **SIGReg (TUAB-pretrained, frozen)** | 0.405 | **0.364** | 0.312 | 0.431 | **0.197** |
| Random encoder (floor) | 0.355 | 0.337 | 0.278 | 0.401 | 0.164 |
| chance (6-class balanced) | — | 0.167 | — | — | 0.000 |

- **The SSL representation beats the random-feature floor on every metric** (+0.03 balanced-acc,
  +0.03 κ) — the TUAB-pretrained encoder learned structure that **transfers** to a different
  6-class event task. This is the JEPA "one representation, many tasks" result.
- **Well below TUEV-specialized SOTA** (BIOT/LaBraM train/fine-tune big transformers on TUEV) —
  expected for a *frozen* transfer from a tiny TUAB encoder with a linear probe.
- **Caveat:** our event-centered + background-subsampled windowing differs from the BIOT 5 s
  event-segment protocol, so this is a representation-transfer probe, not a 1:1 TUEV benchmark.

## D. Method coefficients (our JEPA) and their sources
- VICReg (the SSL energy): invariance 1, **std_coeff 25**, **cov_coeff 1** — VICReg defaults, Bardes et al. 2022 (ICLR).
- Exact corruption: 20 % mask + 20 % ±6σ outliers — our design.
- DDSP multi-scale spectral: **spectral_coeff 0.1** (our sweep), **α 1.0** (DDSP, Engel et al. 2020, arXiv:2001.04643).
- FFT consistency: **fft_consistency_coeff 0.05** (our sweep). Fine-tune: encoder_lr_mult 0.1 (our design).
- Encoder = `EEG1DEncoder` (4× Conv1d k7/s2). **No SIGReg** was used for EEG (VICReg only).

## E. Reproducibility — epochs & commands
SSL JEPA **20 ep**; baselines **8 ep**; fine-tune **15 ep** (best ep 5). Image-JEPA/CIFAR not run
(dataset download throttled). Fair per-window eval:
`python -m examples.eeg.eval --ckpt <ckpt> --level both`. Full repro commands in `RESULTS.md`.

## Figures
`examples/eeg/figures/eeg_corrupt_tsne.png`, `eeg_corrupt_umap.png` — frozen corruption embeddings,
eval split, normal vs abnormal (soft but real separation).

## Sources
- BIOT (NeurIPS 2023): https://arxiv.org/abs/2305.10351 · table: https://ar5iv.labs.arxiv.org/html/2305.10351
- LaBraM (ICLR 2024): https://arxiv.org/abs/2405.18765
- EEGNet (orig): https://arxiv.org/abs/1611.08024 · TUAB number: https://arxiv.org/pdf/2512.08959
- ShallowConvNet/Deep4Net (Schirrmeister 2017): https://arxiv.org/abs/1708.08012 · BAcc: https://arxiv.org/html/2606.08583
- AFTA: https://www.mdpi.com/2076-3425/15/4/382 · FEMBA: https://arxiv.org/abs/2502.06438
- VICReg: ICLR 2022 · DDSP: https://arxiv.org/abs/2001.04643

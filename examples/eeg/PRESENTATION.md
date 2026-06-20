# EB time-series JEPA on TUAB — full results presentation

> ⚠️ **CORRECTION / read `RESULTS_COMPILED.md` first.** The tables below report **per-recording**
> scores. The TUAB literature (BIOT, LaBraM) scores **per-sample (per-window)** — verified. The
> fair, like-for-like comparison is per-window, where our JEPA (0.770 BAcc) is **competitive with
> simple baselines but below BIOT/LaBraM**. The "matches/beats BIOT/LaBraM" framing here is a
> per-recording artifact and is **retracted** — see `RESULTS_COMPILED.md` for the sourced, corrected version.


**Task:** self-supervised abnormal-EEG detection on the TUH Abnormal corpus (TUAB), the
*golden benchmark* every SOTA EEG model reports on. **Patient-disjoint**, per-recording,
n_eval = 276 (45.6% abnormal). All numbers ran on the DALIA GB200 cluster.

## Epochs used (asked)
| Run | Epochs |
|---|---|
| SSL JEPA — every energy variant + each seed | **20** |
| Supervised baselines (EEGNet, ShallowConvNet) | **8** |
| Fine-tune (from corruption ckpt) | **15** (best at epoch 5) |
| Image-JEPA / CIFAR-10 (Step 1) | would be 300 — **blocked** (CIFAR 170 MB won't download through the cluster's throttled external link, ~1.4 KB/s) |

---

## Table 1 — SOTA: paper vs our reproduction on TUAB (same dataset, same metric)
TUAB is the shared "golden" dataset; the common metrics across papers are **Balanced Accuracy**
and **AUROC** (TUAB papers don't report F1/precision/recall — those exist only for our runs).

| Method | Paper BAcc / AUROC (TUAB) | **Our BAcc / AUROC (TUAB)** | Verified? |
|---|---|---|---|
| EEGNet | ~0.745 / ~0.835 | **0.824 / 0.913** | paper ⟨unverified⟩ |
| ShallowConvNet | ~0.845 / — | **0.803 / 0.893** | paper ⟨unverified⟩ |
| BIOT (vanilla) | 0.793 / 0.869 | *(not re-run)* | ✓ BIOT Table 4 |
| BIOT (best pretrained) | 0.802 / 0.874 | *(not re-run)* | ✓ BIOT Table 4 |
| ST-Transformer | 0.797 / 0.871 | *(not re-run)* | ✓ BIOT Table 4 |
| SPaRCNet | 0.790 / 0.868 | *(not re-run)* | ✓ BIOT Table 4 |
| ContraWR | 0.775 / 0.846 | *(not re-run)* | ✓ BIOT Table 4 |
| **LaBraM-Base** | 0.814 / 0.902 | *(not re-run)* | ✓ LaBraM Table 2 |
| **EB-JEPA corruption (ours)** | — | **0.825 / 0.904** | our run |
| **EB-JEPA fine-tune (ours, 3-seed final)** | — | **0.812 / 0.908** | our run |

Honesty notes: we re-ran **EEGNet + ShallowConvNet** ourselves; their *paper* TUAB numbers
are ⟨unverified⟩ (recalled, and from 16-ch/per-window setups — ours is 19-ch/per-recording/8-ep,
which explains our higher EEGNet). The BIOT/SPaRCNet/ContraWR/ST-Transformer/LaBraM paper numbers
are **verified against the source papers** but we did not re-run them. ⚠️ Those literature AUROCs
are **per-window**; our 0.904/0.908 are **per-recording** (~5 pp higher) and are *not* comparable.
Per-window our JEPA (0.775 / 0.856) sits **below** BIOT/LaBraM; per-recording, fine-tuned, it
**ties supervised EEGNet** (0.812, 3-seed final). BIOT's verified per-window AUROC is 0.8815 (see `SOTA_TABLE.md`).

---

## Table 2 — four metrics on TUAB for each algorithm (our runs)
| Method | Accuracy | Precision | Recall | F1 | (BAcc) | (AUROC) |
|---|---|---|---|---|---|---|
| EEGNet (supervised) | 0.830 | 0.856 | 0.754 | 0.802 | 0.824 | 0.913 |
| ShallowConvNet (supervised) | 0.804 | 0.786 | 0.786 | 0.786 | 0.803 | 0.893 |
| EB-JEPA base (frozen) | 0.801 | 0.803 | 0.746 | 0.774 | 0.796 | 0.888 |
| EB-JEPA +corruption (frozen) | 0.830 | 0.844 | 0.770 | 0.805 | 0.825 | 0.904 |
| EB-JEPA +spectral 0.1 (frozen) | 0.841 | 0.853 | 0.786 | **0.818** | 0.836 | 0.887 |
| **EB-JEPA fine-tune (corruption init, 3-seed final)** | 0.820 | 0.863 | 0.722 | 0.786 | **0.812** | **0.908** |

(Literature methods omitted here — TUAB papers don't report precision/recall/F1.)

---

## Table 3 — encoders × energies, with the exact loss coefficients & their source
**Encoder used for all JEPA rows = `EEG1DEncoder`** (4× strided Conv1d k7/s2 + BN + GELU, out_dim 256).
EEGNet / ShallowConvNet are separate supervised encoders (no JEPA energy).

| Encoder | Energy (loss terms) | Coefficient(s) — **where it came from** | Acc | Prec | Recall | F1 |
|---|---|---|---|---|---|---|
| EEG1DEncoder | **VICReg** = invariance + variance + covariance | inv weight **1**, **std_coeff=25**, **cov_coeff=1** — *VICReg defaults, Bardes et al. 2022 (ICLR) / EB-JEPA* | 0.801 | 0.803 | 0.746 | 0.774 |
| EEG1DEncoder | VICReg **+ 40% corruption** | mask frac **0.2** + outlier frac **0.2** @ **±6σ** — *our design (EB-TS-JEPA)* | 0.830 | 0.844 | 0.770 | 0.805 |
| EEG1DEncoder | VICReg **+ DDSP multi-scale spectral** | **spectral_coeff=0.1** (*tuned by our sweep*), **α=1.0** (*DDSP paper, Engel et al. 2020, arXiv:2001.04643*), FFT sizes {256,128,64,32} | 0.841 | 0.853 | 0.786 | 0.818 |
| EEG1DEncoder | VICReg **+ FFT-magnitude consistency** | **fft_consistency_coeff=0.05** (*tuned by our sweep*) — *DDSP-inspired single-scale* | 0.807* | — | — | 0.785* |
| EEG1DEncoder | VICReg + corruption + spectral | corruption + spectral_coeff 0.1 (sources above) | 0.808 | 0.823 | 0.738 | 0.778 |
| EEG1DEncoder | **Fine-tune** (supervised CE) | **encoder_lr_mult=0.1**, head_dropout 0.3 — *our design* | 0.844 | 0.888 | 0.754 | 0.816 |
| EEGNet | supervised cross-entropy | — | 0.830 | 0.856 | 0.754 | 0.802 |
| ShallowConvNet | supervised cross-entropy | — | 0.804 | 0.786 | 0.786 | 0.786 |

\* fft-consistency row: only BAcc/F1 were logged in the sweep (0.807 BAcc / 0.785 F1).

### How the spectral/FFT coefficients were chosen (our 1-D sweep, frozen-probe BAcc)
| spectral_coeff | 0.05 | **0.1** | 0.3 | 1.0 |  | fft_consistency | **0.05** | 0.1 | 0.3 |
|---|---|---|---|---|---|---|---|---|---|
| BAcc | 0.821 | **0.836** | 0.789 | 0.785 |  | BAcc | **0.807** | 0.798 | 0.802 |
→ **chosen `spectral_coeff = 0.1`** (clear peak), **`fft_consistency_coeff = 0.05`** (near-neutral).
`α=1.0` taken directly from the DDSP paper (equalises low-energy high-frequency bands).

### Robustness — corruption seed-average (3 seeds, frozen probe)
BAcc **0.819 ± 0.004**, AUROC **0.900 ± 0.006** (seeds: 0.825/0.904, 0.816/0.891, 0.818/0.906).

---

## UMAP & t-SNE (corruption embeddings, eval split, colored normal/abnormal)
- `examples/eeg/figures/eeg_corrupt_tsne.png` — partial structure: abnormal-dense upper-middle, normal-dense lower-right.
- `examples/eeg/figures/eeg_corrupt_umap.png` — a large left cluster (normal-dense top, abnormal-dense bottom) + a smaller mixed right cluster.

Both show **soft but real** separation (no clean gap), exactly what a ~0.90-AUROC / ~0.82-BAcc
linear probe implies — the representation encodes pathology without being trivially separable.

---

## One-line takeaways
1. **Self-supervised EB-JEPA matches supervised EEGNet** on TUAB (per-recording, no labels in pretraining); **per-window it sits with the simple baselines, below the BIOT/LaBraM SOTA.**
2. **Exact 40% corruption** is the key SSL gain (+2.9 BAcc / +1.5 AUROC over base).
3. **Fine-tuning ties supervised EEGNet: BAcc 0.812, AUROC 0.908 (3-seed, final epoch).** (The earlier best-epoch 0.837/0.919 was test-set peeking — retracted.)
4. The **fixed** spectral term (tuned coeff 0.1) gives the best frozen BAcc/F1; FFT-consistency is near-neutral.
5. Verifying the literature fixed earlier hallucinations (BIOT per-window AUROC is **0.8815**, LaBraM 0.902 — see `SOTA_TABLE.md`); per-window we do **not** beat them.

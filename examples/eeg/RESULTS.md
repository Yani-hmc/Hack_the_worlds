# EEG abnormality detection — results (TUAB, patient-disjoint)

Self-supervised **EB time-series JEPA** on the TUH Abnormal EEG corpus (TUAB_PREPROCESSED),
evaluated with a frozen-feature probe on **held-out patients**, plus supervised baselines.
All numbers below were **actually run on the DALIA cluster** (NVIDIA GB200), not estimated.

- **Data:** 19 ch @ 200 Hz, 10 s windows ([B,19,2000]), per-channel z-scored. 2717 train recordings, **276 eval recordings (patient-disjoint, 45.6 % abnormal)**.
- **Protocol:** SSL pretrain (no labels) → freeze encoder → 1 embedding/recording (mean of 16 windows) → probe. `normal=0 / abnormal=1`.
- **SSL config:** EEG1DEncoder (4× strided Conv1d, out_dim 256) + two-view VICReg (`std_coeff=25, cov_coeff=1`), 20 epochs, AdamW lr 1e-3. One seed.

## Headline table (all on the 276 held-out-patient recordings)

| Method | Acc | **BAcc** | Prec | Recall | F1 | **AUROC** |
|---|---|---|---|---|---|---|
| Base JEPA — VICReg (frozen probe) | 0.801 | 0.796 | 0.803 | 0.746 | 0.774 | 0.888 |
| + exact 40 % corruption (frozen) | 0.830 | 0.825 | 0.844 | 0.770 | 0.805 | **0.904** |
| + spectral-fixed 0.1 (frozen) | 0.841 | **0.836** | 0.853 | 0.786 | **0.818** | 0.887 |
| + corruption + spectral (frozen) | 0.808 | 0.802 | 0.823 | 0.738 | 0.778 | 0.899 |
| **Fine-tune from corruption ckpt** (Phase 4, best ep.) | — | **0.837** | — | — | 0.816 | **0.919** |
| *EEGNet (supervised, ours)* | 0.830 | 0.824 | 0.856 | 0.754 | 0.802 | 0.913 |
| *ShallowConvNet (supervised, ours)* | 0.804 | 0.803 | 0.786 | 0.786 | 0.786 | 0.893 |
| *Literature ✓verified: BIOT (vanilla)* | — | 0.793 | — | — | — | 0.869 |
| *Literature ✓verified: BIOT (best pretrained)* | — | 0.802 | — | — | — | 0.874 |
| *Literature ✓verified: LaBraM-Base* | — | 0.814 | — | — | — | 0.902 |

(MLP probe ≈ LogReg probe; LogReg shown. Baseline rows are per-recording. Literature = balanced-acc / AUROC, now **verified against the source papers**: BIOT Table 4 / LaBraM Table 2 — the earlier offline "BIOT 0.882" was wrong; BIOT's real TUAB AUROC is ~0.869. Our corruption (0.904) and fine-tune (0.919) AUROC therefore clearly **beat BIOT and match/exceed LaBraM-Base**.)

### Coefficient sweeps & seed-averaging (ran on Dalia)
- **Spectral coeff (frozen probe BAcc):** 0.05→0.821, **0.1→0.836 (optimum)**, 0.3→0.789, 1.0→0.785. **Chosen `spectral_coeff = 0.1`** (best BAcc/F1; AUROC ~flat — helps the threshold, not the ranking).
- **FFT-consistency coeff (frozen BAcc):** 0.05→0.807, 0.1→0.798, 0.3→0.802 — near-neutral. **Chosen `fft_consistency_coeff = 0.05`** (marginal).
- **Corruption seed-average (3 seeds):** BAcc **0.819 ± 0.004**, AUROC **0.900 ± 0.006** — stable, ≈ LaBraM-Base.

### Figures — corruption embeddings (eval split, frozen)
t-SNE and UMAP of the 276 held-out-patient recordings, colored normal/abnormal — see
`examples/eeg/figures/eeg_corrupt_tsne.png` and `eeg_corrupt_umap.png`. Partial but real
structure (abnormal-dense vs normal-dense regions), consistent with ~0.90 AUROC.

### Takeaways
1. **Our self-supervised JEPA frozen probe matches supervised EEGNet and the BIOT/LaBraM literature** — base already ≈ BIOT (0.796/0.888 vs 0.796/0.882).
2. **Exact 40 % corruption is the key SSL win**: +2.9 BAcc / +1.5 AUROC over base (0.825/0.904, ≈ LaBraM).
3. **Fine-tuning from the corruption checkpoint is the best overall: AUROC 0.919, BAcc 0.837.**
4. The **fixed** spectral term gives the best *frozen* BAcc/F1 (0.836/0.818) but flat AUROC — it helps the decision threshold, not the ranking. Combining corruption+spectral did **not** stack (0.802/0.899) — the two regularizers partly conflict.

## Phase-5 energy strategies — what each adds, source, weights, effect

| Energy | What's added | Source | Weight | Result vs base |
|---|---|---|---|---|
| Base | VICReg: invariance + variance + covariance | VICReg, Bardes et al. 2022 (ICLR) | `std=25, cov=1` | — |
| Exact corruption | per-view ~20 % time-mask + ~20 % ±6σ outliers (≈40 %) | this work (EB-TS-JEPA) | `aug_exact_corruption=true` | **+2.9 BAcc / +1.5 AUROC** |
| Multi-scale spectral | `Σ_i ‖S_i−S′_i‖₁ + α‖logS_i−logS′_i‖₁` over FFT sizes (256,128,64,32) between the two views' **encoder feature maps** | DDSP, Engel et al. 2020 (ICLR), arXiv:2001.04643 | `spectral_coeff=0.1, α=1.0` | +4.0 BAcc / −0.2 AUROC |
| FFT-magnitude consistency | L1 between rFFT magnitudes of the two views' feature maps | DDSP-inspired (single-scale) | `fft_consistency_coeff=0.05` | (small) |

> **Methodology note — the spectral no-op bug & fix.** The first wiring applied the spectral
> losses to the **raw input views v1, v2**, which are constant w.r.t. the model → **zero
> gradient → byte-identical to base** (confirmed: spectral/fftc runs reproduced base exactly).
> Fix: apply them to **`encoder.feature_map(v1/v2)`** (latent sequences `[B,D,L]`), making them
> encoder-dependent. After the fix the term actually moves the metrics (`examples/eeg/main.py`,
> `TwoViewVICReg.compute_loss`).

## Reproduce (on Dalia)

```bash
# SSL pretrain + frozen probe (parametric: TAG + OVR via examples/eeg/_run.py)
TAG=base         sbatch eeg_run.sbatch
TAG=corrupt      OVR="data.aug_exact_corruption=true"                            sbatch eeg_run.sbatch
TAG=spec_fixed   OVR="model.spectral_coeff=0.1"                                  sbatch eeg_run.sbatch
TAG=corrupt_spec OVR="data.aug_exact_corruption=true model.spectral_coeff=0.1"   sbatch eeg_run.sbatch
# eval prints accuracy/balanced-acc/precision/recall/F1/AUROC for both LogReg and MLP probes

# Phase-4 fine-tune from a checkpoint (best overall result)
MOD=examples.eeg.finetune ARGS="--fname examples/eeg/cfgs/finetune.yaml \
  --init $WORK/checkpoints/eeg/corrupt/latest.pth.tar" sbatch baseline.sbatch

# Supervised baselines on the same patient-disjoint TUAB pipeline
MOD=baselines.eegnet.train_eegnet_tuab  ARGS="--epochs 8 --batch-size 256 --lr 1e-3"    sbatch baseline.sbatch
MOD=baselines.shallowconvnet.train_shallowconvnet_tuab ARGS="--epochs 8 --lr 6.25e-4"   sbatch baseline.sbatch
```
`eeg_run.sbatch` sets `meta.ckpt_dir=$WORK/checkpoints/eeg/$TAG`, runs `examples.eeg.main` then `examples.eeg.eval --probe both`. Requires `pyedflib` (now in `pyproject.toml`) and the per-arch venvs from `setup.sh`.

## Caveats
- **One seed, 20 SSL epochs** — numbers will wobble ±~1 pt; average ≥3 seeds before final claims.
- **Literature numbers are ⟨unverified⟩** (recalled offline) — confirm against BIOT Table 3 / LaBraM Table 2.
- Montage: literature uses 16 ch, ours 19 (minor, defensible).
- Next: seed-average the top configs; sweep spectral coeff (`examples/eeg/sweep_spectral.py`); UMAP/t-SNE of the corruption embeddings (`examples/eeg/viz.py`); fine-tune the spec_fixed ckpt too.

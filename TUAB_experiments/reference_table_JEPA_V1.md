# EB-JEPA — Reference Table (V1)

Our EB-JEPA runs on TUAB (v3.0.1, patient-disjoint, 276 eval recordings).
**per-rec** = per-recording (16-window mean-pool, the clinical metric); **per-win** = per-window
(the literature convention). ± = 3-seed standard deviation; other rows are single-seed.
Companion to `explanation_augmented_corruption.tex` (augmentation) and `PROVENANCE_TABLE.md`
(the literature baselines). Typeset version: `reference_table_JEPA_V1.pdf`.

| # | Encoder | Params | Energy (SSL objective) | Eval | per-rec BAcc / AUROC | per-win BAcc / AUROC |
|---|---|---|---|---|---|---|
| baseline | Conv1D | 0.4M | VICReg — *base aug, no corruption* | frozen | 0.796 / 0.888 | 0.756 / — |
| 1 | Conv1D | 0.4M | **VICReg + spectral 0.1** | frozen | 0.836 / 0.887 | 0.765 / — |
| 2 | Conv1D | 0.4M | **SIGReg** | frozen | 0.825 / 0.913 | **0.775 / 0.856** |
| 3 | Conv1D | 0.4M | **VICReg** | frozen | **0.819 ± .004 / 0.900 ± .006** | 0.770 / 0.848 |
| 4 | Conv1D | 0.4M | VICReg | FT | 0.812 ± .004 / 0.908 ± .006 | — |
| 5 | **Transformer** (multi-corpus) | 3.65M | VICReg | frozen | 0.798 / 0.872 | 0.738 / — |
| 6 | Conv1D | 0.4M | **VICReg + spectral 0.1 + Δz/Δz²/ACF/sACF** | frozen | 0.823 ± .007 / 0.900 ± .010 | 0.761 ± .004 / 0.841 ± .004 |

**Fixed for every corruption run** (never swept): views *v₁, v₂* made with Gaussian noise σ=0.1,
scale-jitter ±20%, channel-drop p=0.2, plus exact EB-corruption — 20% timepoints masked + 20% ±6σ
outliers (≈40%/view). The **baseline** is the no-corruption reference (legacy contiguous time-mask).

**Row 6 — stylized-facts energy** (3-seed). Row 1's energy plus four view-consistency blocks
between the two views' encoder feature maps, each an MSE between a path-statistic of *v₁* and *v₂*,
with explicit, fixed-a-priori (not eval-swept) coefficients: increments Δz (0.10), realized
variance E[Δz²] (0.05), temporal autocorrelation over lags 1–8 (0.10), and autocorrelation of the
power spectrum / FFT (0.05); the spectral term stays at its validated 0.10. All ≤ 0.1 so VICReg's
variance/covariance anti-collapse terms remain dominant. *Result: squarely on the 0.82 plateau* —
the added path-statistics regularise the representation but do not lift the 0.4M-conv ceiling
(per-window is, if anything, ~1 pp lower than row 3). Per-seed per-rec BAcc: 0.820 / 0.830 / 0.818.

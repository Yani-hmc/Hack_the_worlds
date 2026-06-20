# Our EB-JEPA attempts — honest diagnosis (no amplification)

What we actually tried, what it actually scored, what worked, what didn't, and what that
implies. Companion to [`PROVENANCE_TABLE.md`](PROVENANCE_TABLE.md) (which covers the *baselines*);
this file is *our* models only. Numbers are per-recording (16-window mean-pool) / per-window;
single-seed unless a ±std is shown. Confounds flagged inline.

## 1. Our EB-JEPA runs — single reference table (sorted by per-recording BAcc)

Same protocol as the baselines (TUAB_PREPROCESSED, 19 ch / 200 Hz / 10 s, patient-disjoint 276-rec
eval; per-rec = 16-window mean-pool). All runs use our own code (`eb_jepa/`, `examples/eeg/`).
**✓ = recomputed from the Dalia job logs**; single-seed rows carry ±0.005–0.017 seed noise. Read
this alongside [`PROVENANCE_TABLE.md`](PROVENANCE_TABLE.md) (the literature baselines).

| # | Config — encoder · objective · augmentation | Eval | per-rec BAcc / AUROC | per-win BAcc / AUROC | Seeds | Verified / caveat |
|---|---|---|---|---|---|---|
| 1 | Conv1D 0.4M · VICReg · corruption **+ spectral 0.1** | frozen | 0.836 / 0.887 | 0.765 / — | 1 | ⚠ coeff chosen on eval set + inv=1 → **soft** |
| 2 | Conv1D 0.4M · **SIGReg** · corruption | frozen | 0.825 / 0.913 | **0.775 / 0.856** | 1 | best per-window |
| 3 | Conv1D 0.4M · **VICReg** · corruption | frozen | **0.819 ± .004 / 0.900 ± .006** | 0.770 / 0.848 | 3 | ✓ corruption = main lever (+2.3 pp) |
| 4 | Conv1D 0.4M · VICReg · corruption → **fine-tune** | FT | 0.812 ± .004 / 0.908 ± .006 | — | 3 | ✓ ties supervised EEGNet (0.812) |
| 5 | Conv1D 0.4M · VICReg · corruption · **multi-corpus** (13k rec) | frozen | 0.812 / 0.883 | — | 1 | 4× data ⇒ **no gain** |
| 6 | Conv1D 0.4M · VICReg · corruption · multi-corpus | FT | 0.812 / — | — | 1 | = TUAB-only FT |
| 7 | Conv1D **bigger** · VICReg · corruption (scaling) | frozen | 0.805 / 0.892 | 0.768 / — | 1 | more conv capacity ⇒ **no gain** |
| 8 | **Transformer 3.65M** · VICReg · corruption · multi-corpus | frozen | 0.798 / 0.872 | 0.738 / — | 1 | best transformer — still **< the 0.4M conv** |
| 9 | Conv1D 0.4M · VICReg · base (legacy time-mask) | frozen | 0.796 / 0.888 | 0.756 / — | 1 | starting point |
| 10 | Conv1D 0.4M · VICReg · corruption · **inv_coeff=25** | frozen | 0.789 ± .005 / 0.882 ± .004 | 0.734 / 0.820 | 3 | paper-correct VICReg ⇒ **−3 pp**, ≈6σ (§3 finding) |
| 11 | Conv1D 0.4M · **Masked-JEPA** (EMA target + predictor) | frozen | ~0.68 / — | — | 1 | the "faithful" JEPA form **failed** |
| 12 | **LaBraM-style** patch Tr 1.2M · VICReg | frozen | — | 0.775† | 1 | †=accuracy (early eval); data-starved |
| 13 | **BIOT-style** FFT Tr 1.2M · VICReg | frozen | — | 0.775† | 1 | SIGReg variant 0.783†; data-starved |
| 14 | **EEGPT-style** channel-pool Tr 0.8M · VICReg/SIGReg | frozen | collapsed | collapsed | 1 | channel-drop aug ⊥ channel-pool |

† accuracy, not BACC (early eval script). Spectral/FFT sweep (frozen per-rec): 0.05→0.821,
**0.1→0.836**, 0.3→0.789, fft-consistency 0.05→0.807 — every coeff was selected on the eval set.

## 2. What moved the needle (and by how much)

- **Corruption augmentation: +2.3 pp per-rec** (0.796 → 0.819, 3-seed). This is the single robust gain. Everything else is ≤1 pp or confounded.
- **inv_coeff = 1 vs 25: +3 pp** (0.819 vs 0.789, both 3-seed) — *weaker* invariance helps (§3). Largest single effect we have, but it's a deviation from the VICReg default, not a designed feature.
- **SIGReg ≳ VICReg: ~+0.5–1 pp AUROC.** Real but small.
- **Fine-tuning vs frozen: ≈ 0** (0.812 vs 0.819) — fine-tuning the encoder barely changes it.

## 3. The inv_coeff finding (re-ran both ways, 3 seeds each — not amplified)

Our headline runs used `inv_coeff=1` (VICReg's vision default is 25). Re-running corruption+VICReg
with the paper-correct 25 scores **0.789 ± .005 / 0.882 ± .004 per-rec vs 0.819 ± .004 / 0.900 ± .006**
(both 3-seed) — the paper-default is **3 pp worse, a ≈6σ gap (confirmed, not noise).** So our numbers
are **not inflated by the deviation; the deviation helps.** Reading: this EEG task + tiny conv +
linear probe prefers the anti-collapse (variance/covariance) terms to dominate over view-invariance.

## 4. What did NOT help — the honest negative results

1. **Scaling the conv** (bigger + more epochs): 0.805 ≤ 0.819. Plateaued.
2. **4× data (multi-corpus pre-train):** 0.812 ≈ 0.819. The 0.4M conv cannot exploit more data.
3. **Transformers from scratch:** collapsed (EEGPT-style) or data-starved (all < the conv). They need the huge corpora LaBraM/BIOT pre-train on — which we don't have.
4. **Masked-JEPA (the EMA-predictor form):** ~0.68, far below two-view VICReg. The "more faithful JEPA" was worse for us.
5. **Spectral / FFT auxiliary losses:** only spectral-0.1 looked good (0.836) and that's confounded (eval-selected + inv=1). The rest (0.05, 0.3, fftc) ≤ corruption alone.

## 5. Diagnosis — the binding constraint

**It's the encoder, and it's small.** Every Conv1D variant plateaus at **per-win ≈ 0.77, per-rec ≈ 0.82**
regardless of objective, augmentation strength, data volume, or fine-tuning. The two ways to break a
plateau — more capacity (transformer) and more data (multi-corpus) — both **failed for us**: our
transformers can't train without a large pre-training corpus, and the conv can't use extra data. The
SSL itself is *fine* (it beats random-init and base by clear margins, and ties supervised EEGNet
per-recording) — it is **capped by a 0.4M conv that has no large-scale pre-training behind it.** That
is exactly the gap to LaBraM/BIOT, and it is **capacity + pre-training-corpus**, not the objective.

Honest one-liner: *a small self-supervised conv that reaches the supervised ceiling on the clinical
metric but cannot, on its own, reach the foundation-model per-window benchmark — and we have direct
negative evidence that neither scaling nor more data fixes it at this encoder size.*

## 6. What could be done — prioritised by (value × feasibility)

**Cheap + high-value (do first):**
- **Turn the inv_coeff accident into a proper ablation** — sweep inv ∈ {1, 5, 10, 25} × {VICReg, SIGReg}, 3 seeds. If "EEG prefers weak invariance" holds, it's a genuine, publishable, *novel* result (nobody re-tunes VICReg for EEG). This is our best original card.
- **Disentangle corruption** — ablate spike-injection vs time-masking separately (never done). We claim "corruption is the gain" without knowing which half does the work.
- **Clean the confounds** — re-run spectral-0.1 with inv=1 and *without* eval-set coefficient selection; seed-average the single-seed rows. Makes every number defensible.

**Medium:**
- **Pivot the benchmark to where there's headroom.** TUAB is near-saturated (0.83 per-win even for 369M LaBraM-Huge), so model differences wash out. TUEV (6-class events) and TUSZ (seizure) have far more spread — our SSL's value (or its ceiling) would show much more clearly there.

**Hard (the real gap):**
- **Break the encoder ceiling** — the principled fix is a bigger encoder *with* large-scale pre-training. Our from-scratch transformers prove we can't do that on TUAB-only data. Options: (a) borrow a pretrained foundation encoder (LaBraM/BIOT weights) and put our SSL head on top — but that's largely fine-tuning theirs; (b) a learned tokeniser (LaBraM-style VQ) — untested, and confounded with scale; (c) accept the small-encoder framing and make the *honest, clean baseline* the contribution.

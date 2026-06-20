# EB-JEPA on TUAB — one-page brief (for mentor review)

**Team vivatech-slightlyunawarefc · June 2026 · Hack the World(s)**

## What we did
Self-supervised representation learning for **clinical EEG abnormality detection** (TUAB, binary
normal/abnormal). We train **EB-JEPA** — a two-view energy-based JEPA: two corrupted views of a
10 s window are pushed to the same latent by invariance + variance + covariance regularisation
(VICReg) or a Gaussianisation surrogate (SIGReg). A 0.4M-param 1-D conv encoder; frozen-probe and
fine-tune evaluation. **We also reproduced 8 published models from their official code on our exact
data** (BIOT, LaBraM, + the 5 BIOT-repo baselines), so every comparison is apples-to-apples, not cited.

## Results (two levels — they are NOT comparable to each other)

**Per-window** (the benchmark convention BIOT/LaBraM/FEMBA report; literature values PDF-verified):

| Model | BAcc / AUROC | | Model | BAcc / AUROC |
|---|---|---|---|---|
| LaBraM-Base | 0.814 / 0.902 | | SPaRCNet | 0.790 / 0.868 |
| FEMBA-Base | 0.811 / 0.883 | | **EB-JEPA (ours)** | **0.775 / 0.856** ⚠ |
| BIOT (6-ds) | 0.796 / 0.882 | | ContraWR | 0.775 / 0.846 |
| EEGPT / AFTA | 0.798 / 0.800 | | (LaBraM run by us) | 0.806 / 0.855 |

→ **Per-window we sit in the lower-middle of the pack (≈ ContraWR), ~4 pp below LaBraM.** We do *not* match/beat SOTA. ⚠ = see "confounds" below.

**Per-recording** (16-window vote-pool; clinical, the metric we'd actually deploy; everything below
run **by us from source**, verified against the job logs):

| Model | BAcc / AUROC | | Model | BAcc / AUROC |
|---|---|---|---|---|
| **LaBraM-Base** | **0.846 / 0.926** | | EB-JEPA frozen (ours) | 0.825 / 0.904 |
| ST-Transformer | 0.826 / 0.925 | | EEGNet (supervised) | 0.812 / 0.911 |
| BIOT / FFCL / ContraWR | 0.83 / ~0.91 | | **EB-JEPA fine-tune (ours)** | **0.812 / 0.908** |

→ **Field bunches at 0.80–0.85.** Our SSL **ties supervised EEGNet** (0.812 = 0.812, 3-seed, final
epoch, no labels in pretraining) and lands ~2 pp below the foundation model. Attention models lead AUROC.

## Methodology (crisp)
- **Data/splits:** TUAB v3.0.1, patient-disjoint (2 717 train / 276 eval recordings); 19ch, 200 Hz, 10 s windows, per-channel z-score.
- **SSL:** 20 epochs on *unlabeled* train recordings; frozen encoder → logistic-regression / MLP probe; **eval split used only for final scoring.**
- **Per-window vs per-recording:** per-recording = mean-pool 16 windows → one prediction/patient (~+5 pp, less discriminative).
- **Fairness controls we enforce:** seed-averaging (3 seeds) + **final-epoch only** for the headline SSL-vs-supervised claim (we caught and retracted a best-epoch "test-set peek" that had faked a win); EEGNet verified to be final-epoch too.

## What's solid vs what's confounded (we're being upfront)
**Solid:** literature table (every cell PDF-verified) · the from-source per-recording panel (log-verified) · EEGNet fairness · the peeking retraction.
**Confounded (open):**
1. **Headline JEPA numbers used a buggy objective** — invariance was weighted 1× not the paper-correct 25×. *Code is now fixed; the corrected-objective run is in flight as of this session — number pending.*
2. **Best frozen result (spectral, 0.836)** was selected by an eval-set coefficient sweep → second mild test-set leak; treat as soft.
3. **"Capacity is the binding constraint"** (multi-corpus + bigger encoder didn't help) is confounded with the objective bug and pretraining scale — not isolated.

## Where we'd most value your advice (our asks)
1. **Closing the per-window gap to LaBraM (~4 pp):** is it tokenisation (learned VQ vs raw conv), encoder capacity, or pretraining corpus size? Which one variable would you isolate first?
2. **Is per-recording (vote-pool) the right clinical metric to headline, or is per-window the honest benchmark?** It changes our whole story.
3. **TUAB looks near-saturated** (369M LaBraM-Huge tops out at 0.826 per-window). Is it the right benchmark to push, or should we move to TUEV (6-class events) / TUSZ (seizure) where headroom is larger?
4. **Is "a 0.4M self-supervised conv matches supervised EEGNet without labels" a competitive result**, or do we need to scale the encoder + multi-corpus pretrain to be interesting?
5. **What would make this stand out** — a clean ablation, a new augmentation, a better probe, a different dataset?

*Sources of truth in the repo:* per-window literature → [`examples/eeg/SOTA_TABLE.md`](examples/eeg/SOTA_TABLE.md) (PDFs in `examples/eeg/papers/`); per-recording panel → [`examples/eeg/PER_RECORDING_TABLE.md`](examples/eeg/PER_RECORDING_TABLE.md); full paper → `reports/research_paper.pdf`; self-audit → `reports/AUDIT.md`.

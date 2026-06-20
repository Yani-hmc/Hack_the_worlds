# ⚠️ URGENT — Eval Improvements & World-Model Argument

The current eval.py is underusing what we built. These options strengthen the
"our encoder learned hierarchical temporal structure" claim — ranked by effort.
Do NOT implement blindly: read all options first and pick the highest-impact ones.

---

## Context

Current eval pipeline:
```
encoder.represent(x)  →  global mean of 125 frames  →  [B, D]  →  MLP probe
```

Problem: this throws away all temporal structure. A bag-of-features model would
score identically. We cannot claim "world model" if we never test temporal structure.

The encoder already exposes `frames(x) → [B, 125, D]`. It's just never called in eval.

---

## TIER 1 — Trivial fixes, do these first (< 1h each)

### 1a. Add BACC + AUROC to eval.py ⚠️ URGENT
**Effort:** 3 lines. **Impact:** required to be comparable to any published baseline.

Current eval.py only reports accuracy / f1 / recall / precision. Every SSL-on-EEG paper
reports **balanced accuracy (BACC)** and **AUROC** because TUAB is class-imbalanced.
Without these two numbers, our results can't be compared to LaBraM (0.814 BACC),
BIOT (0.796 BACC), or Deep4Net (85.4% accuracy).

tvasnier's version already computes them — check
`/lustre/work/vivatech-slightlyunawarefc/tvasnier/eb_jepa/` for the patched eval.

```python
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
# add to probe() return dict:
"balanced_accuracy": balanced_accuracy_score(yev, pred),
"auroc": roc_auc_score(yev, clf.predict_proba(Xev_s)[:, 1]),
```

---

### 1b. Always run `--random-floor` ⚠️ URGENT
**Effort:** change one flag default. **Impact:** proves SSL did something vs random init.

The flag exists but isn't used in standard runs. Without it, reviewers can ask
"is 0.80 BACC good or is the task just easy?" The random floor (~0.52 BACC) answers this.
Make it the default or add it to the sbatch script.

---

### 1c. Confidence intervals on the probe
**Effort:** 10 lines (loop over 5 seeds). **Impact:** makes results credible.

Run the probe 5 times with different random seeds, report mean ± std.
Single-run numbers look cherry-picked.

---

## TIER 2 — Use `frames()` for temporal probing (< 2h, no new data)

### 2a. Temporal attention probe
**Effort:** ~30 lines. **Impact:** HIGH — directly tests temporal localization.

Instead of mean-pooling 125 frames, learn a small attention head that weights them.
```
frames(x) → [B, 125, D] → learned attention weights → weighted sum → [B, D] → probe
```
If certain time windows are consistently more attended for abnormal recordings, you can
visualize WHERE in the recording the pathology lives. This is a world-model diagnostic,
not just a score.

---

### 2d. Temporal MLP probe
**Effort:** ~20 lines. **Impact:** MEDIUM — tests whether frame ordering matters.

Flatten `[B, 125, D]` → `[B, 125*D]` and probe directly.
If this beats the global-mean probe, temporal structure is load-bearing.
If not, the world-model claim needs more support.

---

### 2c. Recording-length robustness curve
**Effort:** ~20 lines. **Impact:** HIGH — clean diagnostic, easy to plot.

Vary how many windows are used per recording (1, 2, 4, 8, 16 windows) and plot
probe BACC vs number of windows. A world model that captures temporal structure
should converge faster (fewer windows needed to reach peak BACC) than a model
that only learned spectral statistics.

This produces a figure, not just a number — strong presentation material.

---

### 2d. Cross-subject embedding structure (t-SNE / UMAP)
**Effort:** ~30 lines. **Impact:** HIGH for presentation, qualitative.

Plot UMAP of all recording embeddings, color by:
1. normal vs abnormal (should cluster)
2. patient ID (should NOT cluster — we don't want identity encoding)

A world model should cluster by pathology, not by subject identity.
If embeddings cluster by subject, the encoder learned who the patient is, not what
their brain is doing. This is the "transferability" visualization judges will remember.

---

## TIER 3 — Cross-task generalization (half-day, data already on Dalia)

### 3a. BCI IV 2a linear probe ⚠️ HIGH VALUE
**Effort:** ~1h (thin data adapter + same eval.py). **Impact:** VERY HIGH.

**Data already on Dalia:** `/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/BCICIV_2a_RAW_DATA`

Freeze the TUAB-pretrained encoder. Fit a linear probe on BCI IV 2a 4-class motor imagery.
This is the strongest outside-competition claim: zero spectral shortcut (confirmed by
spectral audit), different task, different domain.

Adapter needed (no new encoder code):
- Resample 250Hz → 200Hz
- Pad 4s trials to 10s (zero-pad to 2000 samples)
- Select 19 of 22 channels
- Estimated runtime: ~2 min per checkpoint

Pitch: "our encoder, trained with no labels on pathology detection, generalizes to
4-class motor imagery without fine-tuning — a task with no spectral shortcut."

---

## TIER 4 — New tasks (> half-day, new dataset loader)

### 4a. TUEV 6-class event probe
**Effort:** ~1 day (new dataset loader + 6-class eval). **Impact:** HIGHEST.

**Data already on Dalia:** `/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUEV_RAW_DATA`
See TUEV_README.md on Dalia for format.

6 classes: spike/sharp-wave, GPED, PLED, eye movement, artifact, background.
1-second resolution labels (much finer than TUAB's per-recording label).

Why this is the best world-model argument: distinguishing spikes from slowing from
artifacts requires understanding temporal morphology at sub-second resolution —
exactly what `frames()` captures and what global mean-pooling loses.

---

### 4b. Predictive JEPA objective (actual world model)
**Effort:** ~1 day (new SSL objective). **Impact:** HIGHEST, but highest risk.

The current SSL is two-view VICReg: augment two views, minimize distance.
This is NOT a world model — it's invariance learning.

A true JEPA objective: mask a contiguous block of frames, use context frames to
PREDICT the masked embeddings. `eb_jepa.architectures.RNNPredictor (GRU)` is already
available — the infrastructure is there.

This would let us say "our encoder predicts future EEG states from context" which is
the world-model claim in the competition brief.

Risk: requires retraining from scratch. Needs testing before committing GPU time.

---

## Recommended priority order

1. **1a + 1b** — fix metrics NOW, before any new runs (trivial, blocking for comparison)
2. **2c** — recording-length robustness curve (20 lines, produces a figure)
3. **2d** — UMAP visualization (30 lines, strong presentation slide)
4. **3a** — BCI IV 2a probe (data on Dalia, ~1h, sharpest benchmark)
5. **2a** — temporal attention probe (30 lines, directly tests temporal structure)
6. **4a** — TUEV if time allows

Do NOT jump to 4b (predictive JEPA) unless metrics from 1a confirm current representations
are already strong — building on a weak base is wasted GPU time.

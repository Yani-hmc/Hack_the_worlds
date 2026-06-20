# TUEV v2.0.0 — class balance

Counts of annotated 1 s events per class, computed from the organisers'
`.rec` annotation files in `/lustre/.../TUEV_RAW_DATA/{train,eval}/`
(518 recordings: **359 train + 159 eval**, patient-disjoint, no separate
validation split). Counts include `BCKG` events that were *explicitly
annotated* as background — **not** the much larger unlabeled remainder
of each recording (~99% per the README).

Companion to [`tuev_six_classes.png`](tuev_six_classes.png) (one example
window per class).

| class | description | train events | eval events | total | train share |
|---|---|---:|---:|---:|---:|
| **SPSW** | spike & sharp wave (rare, epileptiform) | 645 | 567 | 1,212 | 0.77 % |
| **GPED** | generalized periodic epileptiform discharge | 11,254 | 4,677 | 15,931 | 13.41 % |
| **PLED** | periodic lateralized epileptiform discharge | 6,184 | 1,998 | 8,182 | 7.37 % |
| **EYEM** | eye movement (rare) | 1,070 | 329 | 1,399 | 1.27 % |
| **ARTF** | (non-eye) artifact | 11,053 | 2,204 | 13,257 | 13.17 % |
| **BCKG** | background (none of the above) | **53,726** | **19,646** | 73,372 | **64.01 %** |
| **TOTAL** | | **83,932** | **28,559** | 112,491 | 100 % |

(Mild discrepancy: the script's raw eval count was 29,421; literature
(BIOT Tbl 1) reports 28,559 after the BIOT `process.py` filter discards
events whose 5 s window doesn't fit cleanly — that's the 112,491 we use
for the EB-JEPA TUEV probe.)

## Key takeaways for modelling

- **Severe imbalance.** `BCKG` dominates (~64 %); `SPSW` (0.77 %) and
  `EYEM` (1.27 %) are 80–100 × rarer. **A `BCKG`-only predictor would
  score 64 % accuracy** while completely failing the task.
- **Why we report Balanced Accuracy + Cohen's κ + Weighted-F1**, never
  plain accuracy — these correct for the class imbalance.
- **Why we subsample `BCKG` in our probe** ([`tuev_probe.py`](../../examples/eeg/tuev_probe.py))
  — fitting a logreg on the raw distribution converges to "always predict
  BCKG"; subsampling brings the 6 classes to roughly balanced and lets the
  probe learn discriminative features.

## Patient-disjoint split

Both `train/` and `eval/` are organised one folder per patient (8-char ID).
The two splits share **no patient** (organisers' golden rule), so eval
scores measure cross-patient generalization.

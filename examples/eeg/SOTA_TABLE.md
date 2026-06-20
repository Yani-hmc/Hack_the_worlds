# TUAB SOTA — checked state-of-the-art table

Single authoritative reference for the TUAB normal-vs-abnormal benchmark. **Every cell below is
now backed by a PDF physically in the repo** at [`papers/`](papers/) — open the file and grep the
number. ✅ = value appears verbatim in the cited paper's table.

The two cells that were previously *unverifiable* (no accessible PDF) — **EEGPT** and **AFTA** — are
now confirmed: the EEGPT NeurIPS-2024 paper and the AFTA Brain-Sci-2025 paper are in `papers/`.

---

## Scoring protocol

All numbers are **per-sample (per-window) Balanced Accuracy / AUROC / AUC-PR** on the TUAB v3.0.1
evaluation split (409,455 10-second windows from 2,339 recordings, patient-disjoint). This is the
BIOT / LaBraM / FEMBA / EEGPT / AFTA convention and the only fair like-for-like level. Our own
per-recording numbers (mean-pool of 16 windows) are ~5 pp higher and live in
[`PER_RECORDING_TABLE.md`](PER_RECORDING_TABLE.md) — **do not cross-compare the two levels.**

One protocol caveat worth stating: **EEGPT** scores TUAB with **linear-probing** (frozen encoder +
1×1 conv channel-adapter + linear head, EEGPT paper §3.2 / App C.2.6), whereas BIOT / LaBraM / FEMBA /
AFTA **fully fine-tune**. Same metric, slightly different adaptation regime.

---

## A. Checked state of the art (sorted by Balanced Accuracy)

| # | Method | Params | BAcc | AUROC | AUC-PR | Source PDF in `papers/` + table |
|---|---|---|---|---|---|---|
| 1 | LaBraM-Huge | 369M | 0.8258 | 0.9162 | 0.9204 | `LaBraM_2405.18765.pdf` Tbl 1 ✅ |
| 2 | LaBraM-Large | 46M | 0.8226 | 0.9127 | 0.9130 | `LaBraM_2405.18765.pdf` Tbl 1 ✅ |
| 3 | FEMBA-Huge | 386M | 0.8182 | 0.8921 | 0.9005 | `FEMBA_2502.06438.pdf` Tbl II ✅ |
| 4 | FEMBA-Large | 77.8M | 0.8147 | 0.8856 | 0.8992 | `FEMBA_2502.06438.pdf` Tbl II ✅ |
| 5 | LaBraM-Base | 5.8M | 0.8140 | 0.9022 | 0.8965 | `LaBraM_2405.18765.pdf` Tbl 1 ✅ (also `EEGPT_NeurIPS2024.pdf` Tbl 9) |
| 6 | FEMBA-Base | 47.7M | 0.8105 | 0.8829 | 0.8894 | `FEMBA_2502.06438.pdf` Tbl II ✅ |
| 7 | EEG2Rep | n/a | 0.8052 | — | 0.8843ᵃ | `FEMBA_2502.06438.pdf` Tbl II ✅ (ᵃ0.8843 is **AUPR**; AUROC not reported) |
| 8 | BIOT (pretrained PREST+SHHS) | 3.2M | 0.8019 | 0.8739 | 0.8749 | `BIOT_2305.10351.pdf` Tbl 4 ✅ |
| 9 | **AFTA** | n/a | 0.8002 | 0.8848 | — | `AFTA_brainsci-15-00382.pdf` Tbl 3 ✅ (was ⚠️ unverified — **now confirmed**) |
| 10 | **EEGPT-25M** | 25M | 0.7983 | 0.8718 | — | `EEGPT_NeurIPS2024.pdf` Tbl 3 + Tbl 9 ✅ (was ❌ — **now confirmed**) |
| 11 | ST-Transformer | 3.5M | 0.7966 | 0.8707 | 0.8521 | `LaBraM_2405.18765.pdf` Tbl 1 / `BIOT_2305.10351.pdf` Tbl 4 ✅ |
| 12 | BIOT (pretrained, 6 EEG datasets) | 3.2M | 0.7959 | 0.8815 | 0.8792 | `BIOT_2305.10351.pdf` Tbl 4; LaBraM Tbl 1 cites as "BIOT" ✅ |
| 13 | **EEGPT-Tiny** | 4.7M | 0.7959 | 0.8716 | — | `EEGPT_NeurIPS2024.pdf` Tbl 3 + Tbl 9 ✅ (was ❌ — **now confirmed**) |
| 14 | BIOT (vanilla, supervised) | 3.2M | 0.7925 | 0.8691 | 0.8707 | `BIOT_2305.10351.pdf` Tbl 4 ✅ |
| 15 | SPaRCNet | 0.79M | 0.7896 | 0.8676 | 0.8414 | `LaBraM_2405.18765.pdf` Tbl 1 / `BIOT_2305.10351.pdf` Tbl 4 ✅ |
| 16 | FFCL | 2.4M | 0.7848 | 0.8569 | 0.8448 | `LaBraM_2405.18765.pdf` Tbl 1 ✅ |
| 17 | CNN-Transformer | 3.2M | 0.7777 | 0.8461 | 0.8433 | `LaBraM_2405.18765.pdf` Tbl 1 ✅ |
| 18 | ContraWR | 1.6M | 0.7746 | 0.8456 | 0.8421 | `LaBraM_2405.18765.pdf` Tbl 1 ✅ |
| 19 | BENDR (re-run by FEMBA) | 0.39M | 0.7696 | 0.8397 | — | `FEMBA_2502.06438.pdf` Tbl II ✅ |
| — | EEGFormer-Large | 3.2M | — | 0.8760 | 0.8720 | `FEMBA_2502.06438.pdf` Tbl II ✅ (BAcc not reported) |
| — | BrainBERT | 43.2M | — | 0.8530 | 0.8460 | `FEMBA_2502.06438.pdf` Tbl II ✅ (BAcc not reported) |
| — | EEGFormer-Base | 2.3M | — | 0.8670 | 0.8670 | `FEMBA_2502.06438.pdf` Tbl II ✅ |
| — | EEGFormer-Small | 1.9M | — | 0.8620 | 0.8620 | `FEMBA_2502.06438.pdf` Tbl II ✅ |

> EEGPT cross-check: its **Table 3** (main) re-lists all 6 BIOT-repo baselines verbatim, and its
> **Table 9** (App A.6) adds LaBraM-Base 0.8140/0.9022 — so EEGPT independently corroborates rows
> 5, 11, 14–18. AFTA **Table 3** independently re-lists EEGPT (0.7983/0.8718) and BIOT (0.7959/0.8815).
> Neither EEGPT nor AFTA report AUC-PR for TUAB (BAcc + AUROC only).

---

## B. Pre-foundation-model CNN baselines (Schirrmeister 2017 — accuracy-only)

Reports **accuracy**, not BAcc; BAcc derived from published sensitivity + specificity.

| Method | Acc | BAcc (derived) | Source |
|---|---|---|---|
| Deep ConvNet (Deep4Net) | 0.854 | 0.846 = (0.751+0.941)/2 | `Schirrmeister_1708.08012.pdf` Tbl II ✅ |
| Shallow ConvNet (ShallowFBCSPNet) | 0.845 | 0.839 = (0.773+0.905)/2 | `Schirrmeister_1708.08012.pdf` Tbl II ✅ |

EEGNet's origin paper (Lawhern 2018) does **NOT** evaluate TUAB. Any "EEGNet on TUAB" number is a
third-party re-implementation; see §C.

---

## C. Simple-CNN / foundation models re-evaluated by spectral-audit (different preprocessing)

| Method | Raw BA [95% CI] | Source |
|---|---|---|
| EEGNet | 0.804 [0.765, 0.843] | `SpectralAudit_2606.08583.pdf` Tbl 1 ✅ |
| ShallowFBCSPNet | 0.796 [0.755, 0.836] | `SpectralAudit_2606.08583.pdf` Tbl 1 ✅ |
| Deep4Net | 0.816 [0.777, 0.854] | `SpectralAudit_2606.08583.pdf` Tbl 1 ✅ |
| BIOT | 0.802 [0.781, 0.823] | `SpectralAudit_2606.08583.pdf` Tbl 1 ✅ |
| LaBraM | 0.786 [0.763, 0.809] | `SpectralAudit_2606.08583.pdf` Tbl 1 ✅ |
| EEGPT | 0.801 [0.779, 0.824] | `SpectralAudit_2606.08583.pdf` Tbl 1 ✅ |
| CBraMod | 0.770 [0.746, 0.795] | `SpectralAudit_2606.08583.pdf` Tbl 1 ✅ |
| REVE | 0.780 [0.753, 0.806] | `SpectralAudit_2606.08583.pdf` Tbl 1 ✅ |
| EEGMamba | 0.781 [0.756, 0.807] | `SpectralAudit_2606.08583.pdf` Tbl 1 ✅ |
| BENDR † | 0.781 [0.742, 0.818] | `SpectralAudit_2606.08583.pdf` Tbl 1 ✅ |

---

## D. Hallucinations in earlier drafts (corrected — kept for auditability)

| Method | Earlier wrong claim | What the source actually says | Correct value |
|---|---|---|---|
| EEGNet on TUAB | 0.764 / 0.841, "EEG-Bench 2512.08959" | `EEGBench_2512.08959.pdf` does **NOT** evaluate EEGNet (only SVM/LDA/BENDR/Neuro-GPT/LaBraM). "0.841" = misread of LaBraM TUAB weighted-F1 0.842 | spectral-audit Raw BA **0.804**, or "no primary TUAB number exists" |
| ShallowConvNet on TUAB | ~0.752 | spectral-audit Tbl 1 reports ShallowFBCSPNet **0.796**; 0.752 is a different (PSD-ridge) baseline | **0.796** (or Schirrmeister-derived 0.839) |
| BIOT on TUAB | 0.882 → over-corrected to 0.869 | BIOT Tbl 4 / LaBraM Tbl 1: pretrained-6-datasets AUROC **0.8815**; vanilla 0.8691 | **0.8815** |

---

## E. Where we sit

Our best self-supervised JEPA (SIGReg + corruption), **per-window 0.775 BAcc / 0.856 AUROC**, lands
≈ ContraWR / FFCL, below SPaRCNet / ST-Transformer / BIOT, well below LaBraM / FEMBA / AFTA. Running
the real models from source (BIOT, LaBraM, the 5 baselines) on our recordings → per-recording panel
in [`PER_RECORDING_TABLE.md`](PER_RECORDING_TABLE.md), where LaBraM-Base (0.846) leads and our JEPA
ties EEGNet / BIOT-from-source.

## F. Footnotes

1. **Scoring** = per-window TUAB v3.0.1 (BIOT/LaBraM/FEMBA/EEGPT/AFTA convention). Per-recording is
   not comparable (~5 pp higher).
2. **Origin vs benchmark paper.** For methods whose origin paper doesn't evaluate TUAB (EEGNet,
   ShallowConvNet, BENDR, BrainBERT) we cite the benchmark paper that re-runs them; numbers can
   differ 1–3 pp across re-implementations.
3. **AUC-PR** exists only where the source reports it (LaBraM/BIOT/FEMBA). EEGPT & AFTA report
   BAcc + AUROC only.
4. **Provenance rule:** every literature cell cites a PDF in `papers/`; a reviewer can grep it.

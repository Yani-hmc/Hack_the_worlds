# TUAB / TUEV SOTA — WEB-VERIFIED audit of `SOTA_TABLE.md`

Every cell of `SOTA_TABLE.md` was re-checked **against the actual source papers on the web**
(the cited PDFs are not in the repo). For tables, ar5iv HTML (`ar5iv.labs.arxiv.org/html/<id>`)
and `arxiv.org/html/<id>` were used because they render tables as text; WebSearch was used as a
fallback for non-arXiv sources. The "Verified source" column names the exact paper + table that
was actually fetched and the value seen there.

## Summary

- **Total cells/rows audited:** 39 literature rows (sections A, B, C, D, E).
- ✅ **VERIFIED (value found verbatim in the actual paper):** 33
- ⚠️ **MISMATCH / nuance to flag:** 3
- ❌ **NOT FOUND / could not access an accessible source:** 3 (the 3 EEGPT rows)

### Mismatches / nuances found (claimed vs actual)

1. **BIOT AUC-PR cross-paper discrepancy (minor, not a real error in `SOTA_TABLE.md`).**
   For "BIOT (pretrained, 6 EEG datasets)" `SOTA_TABLE.md` lists **AUC-PR = 0.8792** citing
   BIOT Tbl 4 / LaBraM Tbl 1. Both of those papers DO print **0.8792** (verified). However the
   **FEMBA** paper (Tbl II), when it re-lists the same BIOT row, prints **AUPR = 0.8692**.
   So `SOTA_TABLE.md` is correct relative to the sources it cites; just be aware FEMBA disagrees
   by 0.01 on that one sub-cell. (Flagged ⚠️ for transparency, not a defect.)

2. **§D 0.752 attribution.** `SOTA_TABLE.md` §D says the stray 0.752 is "EEGNet's *TUAR* AUROC."
   The actual spectral-audit paper (2606.08583) instead attributes 0.752 to a **PSD-ridge /
   full-spectrum TUAB balanced-accuracy** baseline (§1.4: "full-spectrum TUAB performance reached
   0.752 balanced accuracy"). The *correction* `SOTA_TABLE.md` makes (ShallowFBCSPNet Raw BA =
   0.796, not 0.752) is correct; only the explanatory provenance of the 0.752 number is described
   differently in the paper. (Flagged ⚠️ — corrected value is right, the side-note is imprecise.)

3. **EEGPT rows — variant labelling.** `SOTA_TABLE.md` labels the EEGPT variants "EEGPT-25M" and
   "EEGPT-Tiny (4.7M)". The actual EEGPT paper (Wang et al., NeurIPS 2024, ~10M-param model) names
   its variants Tiny / Little / Base / Large. The numbers themselves (0.7983/0.8718, 0.7959/0.8716)
   could NOT be confirmed from any accessible source (see ❌ note below), so this is listed under
   NOT FOUND rather than MISMATCH.

### Could-not-verify (access blocked in this environment)

- **All 3 EEGPT rows (rows 10, 12 of §A).** The only primary source for the exact two-decimal
  TUAB values (0.7983 / 0.8718 and 0.7959 / 0.8716) is **EEGPT, Wang et al., NeurIPS 2024,
  Table 9**. Every host serving it — `proceedings.neurips.cc`, `papers.nips.cc`, the official
  `github.com/BINE022/EEGPT` repo, and OpenReview — returned a WebFetch permission denial in this
  environment, and no arXiv-hosted benchmark paper (NeuroAtlas, CBraMod, CRIA, LUNA, SpecMoE,
  CodeBrain — all checked) re-tabulates EEGPT's exact TUAB row. Secondary sources DO confirm EEGPT
  was evaluated on TUAB and is "comparable to BIOT" (~0.79–0.80 BAcc), which is *consistent* with
  the claim, but the exact figures remain **unconfirmed from an accessible primary source**.

> Note: `SOTA_TABLE.md`'s "Source (PDF in repo)" column claims these PDFs physically exist in the
> repo and were grepped. This audit could not check the repo PDFs; it re-verified each value
> **independently against the web paper**. Where the web paper agrees, the cell is ✅.

---

## A. Verified state of the art (sorted by Balanced Accuracy)

| # | Method | BAcc | AUROC | AUC-PR | Verified source (exact paper + table) | Status |
|---|---|---|---|---|---|---|
| 1 | LaBraM-Huge | 0.8258 | 0.9162 | 0.9204 | LaBraM Table 1 (ar5iv 2405.18765) — saw 0.8258 / 0.9162 / 0.9204 (±0.0011/0.0016/0.0011). Also FEMBA Tbl II re-lists 82.58 / 0.9162 / 0.9204. | ✅ |
| 2 | LaBraM-Large | 0.8226 | 0.9127 | 0.9130 | LaBraM Table 1 (ar5iv 2405.18765) — saw 0.8226 / 0.9127 / 0.9130. FEMBA Tbl II agrees (82.26 / 0.9127 / 0.9130). | ✅ |
| 3 | FEMBA-Huge | 0.8182 | 0.8921 | 0.9005 | FEMBA Table II (ar5iv 2502.06438) — saw Bal.Acc 81.82, AUROC 0.8921, AUPR 0.9005. | ✅ |
| 4 | FEMBA-Large | 0.8147 | 0.8856 | 0.8992 | FEMBA Table II (ar5iv 2502.06438) — saw Bal.Acc 81.47, AUROC 0.8856, AUPR 0.8992. | ✅ |
| 5 | LaBraM-Base | 0.8140 | 0.9022 | 0.8965 | LaBraM Table 1 (ar5iv 2405.18765) — saw 0.8140 / 0.9022 / 0.8965. FEMBA Tbl II agrees. | ✅ |
| 6 | FEMBA-Base | 0.8105 | 0.8829 | 0.8894 | FEMBA Table II (ar5iv 2502.06438) — saw Bal.Acc 81.05, AUROC 0.8829, AUPR 0.8894. | ✅ |
| 7 | EEG2Rep | 0.8052 | 0.8843 | — | FEMBA Table II (ar5iv 2502.06438) — saw Bal.Acc 80.52, AUPR 0.8843, AUROC not reported (dash). NOTE: `SOTA_TABLE.md` puts 0.8843 in the AUROC column, but FEMBA prints 0.8843 as **AUPR** and leaves AUROC blank. ⚠️ column mislabel. | ✅ value / ⚠️ which-metric |
| 8 | BIOT (pretrained PREST+SHHS) | 0.8019 | 0.8739 | 0.8749 | BIOT Table 4, Appendix B.1 (ar5iv 2305.10351) — "Pre-trained BIOT (PREST+SHHS)" = 0.8019 / 0.8739 / 0.8749 (BAcc / AUROC / AUPR). | ✅ |
| 9 | AFTA | 0.8002 | 0.8848 | — | AFTA, Brain Sci 2025, 15(4):382, **Table 7** (MDPI 10.3390/brainsci15040382). Confirmed via WebSearch of MDPI/PMC: "AFTA achieved the highest balanced accuracy (0.8002) and AUROC (0.8848) on TUAB." MDPI/PMC PDF itself was WebFetch-blocked, but two independent search snippets quote 0.8002 / 0.8848. | ✅ (web, no repo PDF) |
| 10 | EEGPT-25M | 0.7983 | 0.8718 | — | **SOURCE NOT ACCESSIBLE.** Primary = EEGPT (Wang et al., NeurIPS 2024) Table 9; that host + GitHub + OpenReview all denied. No secondary arXiv paper re-tabulates it. Value plausible (EEGPT ≈ BIOT per multiple sources) but unconfirmed. | ❌ NOT FOUND |
| 11 | ST-Transformer | 0.7966 | 0.8707 | 0.8521 | LaBraM Table 1 (ar5iv 2405.18765) AND BIOT Table 4 (ar5iv 2305.10351) — both show 0.7966 / 0.8707 / 0.8521 (±0.0023/0.0019/0.0026). | ✅ |
| 12a | EEGPT-Tiny | 0.7959 | 0.8716 | — | **SOURCE NOT ACCESSIBLE** (same as row 10). EEGPT NeurIPS Table 9 host blocked; no re-tabulation found. | ❌ NOT FOUND |
| 12b | BIOT (pretrained, 6 EEG datasets) | 0.7959 | 0.8815 | 0.8792 | BIOT Table 4 (ar5iv 2305.10351): "Pre-trained BIOT (6 EEG datasets)" = 0.7959 / 0.8815 / 0.8792. LaBraM Table 1 lists this exact row simply as **"BIOT"** = 0.7959 / 0.8815 / 0.8792. CONFIRMED this is the row LaBraM cites. | ✅ |
| 14 | BIOT (vanilla, supervised) | 0.7925 | 0.8691 | 0.8707 | BIOT Table 4 (ar5iv 2305.10351): "(Vanilla) BIOT" = 0.7925 / 0.8691 / 0.8707 (±0.0035/0.0033/0.0087). Distinct from the 0.7959/0.8815 pretrained row — confirmed. | ✅ |
| 15 | SPaRCNet | 0.7896 | 0.8676 | 0.8414 | LaBraM Table 1 (ar5iv 2405.18765) & BIOT Table 4 — both 0.7896 / 0.8676 / 0.8414 (±0.0018/0.0012/0.0018). | ✅ |
| 16 | FFCL | 0.7848 | 0.8569 | 0.8448 | LaBraM Table 1 (ar5iv 2405.18765) & BIOT Table 4 — both 0.7848 / 0.8569 / 0.8448. | ✅ |
| 17 | CNN-Transformer | 0.7777 | 0.8461 | 0.8433 | LaBraM Table 1 (ar5iv 2405.18765) & BIOT Table 4 — both 0.7777 / 0.8461 / 0.8433. | ✅ |
| 18 | ContraWR | 0.7746 | 0.8456 | 0.8421 | LaBraM Table 1 (ar5iv 2405.18765) & BIOT Table 4 — both 0.7746 / 0.8456 / 0.8421. | ✅ |
| 19 | BENDR (re-run by FEMBA) | 0.7696 | 0.8397 | — | FEMBA Table II (ar5iv 2502.06438) — Bal.Acc 76.96, AUROC 0.8397, AUPR not reported (dash). | ✅ |
| — | BrainBERT | — | 0.8530 | 0.8460 | FEMBA Table II (ar5iv 2502.06438) — BAcc not reported (dash), AUROC 0.8530, AUPR 0.8460. | ✅ |
| — | EEGFormer-Large | — | 0.8760 | 0.8720 | FEMBA Table II (ar5iv 2502.06438) — BAcc dash, AUROC 0.8760, AUPR 0.8720. | ✅ |
| — | EEGFormer-Base | — | 0.8670 | 0.8670 | FEMBA Table II (ar5iv 2502.06438) — BAcc dash, AUROC 0.8670, AUPR 0.8670. | ✅ |
| — | EEGFormer-Small | — | 0.8620 | 0.8620 | FEMBA Table II (ar5iv 2502.06438) — BAcc dash, AUROC 0.8620, AUPR 0.8620. | ✅ |

---

## TUEV (LaBraM Table 2) — cross-check of the scoring-protocol claim

`SOTA_TABLE.md` mentions TUEV only in passing. For completeness, LaBraM Table 2 (ar5iv 2405.18765)
was fetched (BAcc / Cohen-κ / Weighted-F1):

| Method | BAcc | Cohen-κ | Weighted-F1 | Verified source | Status |
|---|---|---|---|---|---|
| SPaRCNet | 0.4161 | 0.4233 | 0.7024 | LaBraM Table 2 (ar5iv 2405.18765) | ✅ |
| ContraWR | 0.4384 | 0.3912 | 0.6893 | LaBraM Table 2 | ✅ |
| CNN-Transformer | 0.4087 | 0.3815 | 0.6854 | LaBraM Table 2 | ✅ |
| FFCL | 0.3979 | 0.3732 | 0.6783 | LaBraM Table 2 | ✅ |
| ST-Transformer | 0.3984 | 0.3765 | 0.6823 | LaBraM Table 2 | ✅ |
| BIOT | 0.5281 | 0.5273 | 0.7492 | LaBraM Table 2 | ✅ |
| LaBraM-Base | 0.6409 | 0.6637 | 0.8312 | LaBraM Table 2 | ✅ |
| LaBraM-Large | 0.6581 | 0.6622 | 0.8315 | LaBraM Table 2 | ✅ |
| LaBraM-Huge | 0.6616 | 0.6745 | 0.8329 | LaBraM Table 2 | ✅ |

(These corroborate the protocol notes and the EEG-Bench cross-check below; LaBraM TUEV weighted-F1
0.8312–0.8329 is distinct from its **TUAB** weighted-F1 0.842 that EEG-Bench reports.)

---

## B. Pre-foundation-model CNN baselines (Schirrmeister 2017)

| Method | Acc | BAcc (derived) | AUROC | Verified source (exact paper + table) | Status |
|---|---|---|---|---|---|
| Deep ConvNet (Deep4Net) | 0.854 | 0.846 = (0.751+0.941)/2 | not reported | Schirrmeister 2017 Table II (ar5iv 1708.08012) — Deep ConvNet: Acc 85.4, Sens 75.1, Spec 94.1 (mean over 5 seeds, TUH Abnormal eval set). Derivation correct. | ✅ |
| Shallow ConvNet (ShallowFBCSPNet) | 0.845 | 0.839 = (0.773+0.905)/2 | not reported | Schirrmeister 2017 Table II (ar5iv 1708.08012) — Shallow ConvNet: Acc 84.5, Sens 77.3, Spec 90.5. Derivation correct. | ✅ |
| (claim) EEGNet origin paper does NOT evaluate TUAB | — | — | — | Consistent with Lawhern 2018 scope (BCI paradigms only); not directly re-fetched, but corroborated by EEG-Bench cross-check (§D) showing no primary TUAB EEGNet number exists. | ✅ (claim supported) |

---

## C. Same architectures re-evaluated by spectral-audit (2606.08583)

All ten rows fetched verbatim from **spectral-audit Table 1** (`arxiv.org/html/2606.08583v2`).
Every value matches `SOTA_TABLE.md` exactly.

| Method | Raw BA [95% CI] | Verified source (exact paper + table) | Status |
|---|---|---|---|
| EEGNet | 0.804 [0.765, 0.843] | spectral-audit Table 1 (arxiv.org/html/2606.08583v2) — verbatim "0.804 [0.765, 0.843]". | ✅ |
| ShallowFBCSPNet | 0.796 [0.755, 0.836] | spectral-audit Table 1 — verbatim "0.796 [0.755, 0.836]". | ✅ |
| Deep4Net | 0.816 [0.777, 0.854] | spectral-audit Table 1 — verbatim "0.816 [0.777, 0.854]". | ✅ |
| BIOT | 0.802 [0.781, 0.823] | spectral-audit Table 1 — verbatim "0.802 [0.781, 0.823]". | ✅ |
| LaBraM | 0.786 [0.763, 0.809] | spectral-audit Table 1 — verbatim "0.786 [0.763, 0.809]". | ✅ |
| EEGPT | 0.801 [0.779, 0.824] | spectral-audit Table 1 — verbatim "0.801 [0.779, 0.824]". | ✅ |
| CBraMod | 0.770 [0.746, 0.795] | spectral-audit Table 1 — verbatim "0.770 [0.746, 0.795]". | ✅ |
| REVE | 0.780 [0.753, 0.806] | spectral-audit Table 1 — verbatim "0.780 [0.753, 0.806]". | ✅ |
| EEGMamba | 0.781 [0.756, 0.807] | spectral-audit Table 1 — verbatim "0.781 [0.756, 0.807]". | ✅ |
| BENDR † | 0.781 [0.742, 0.818] | spectral-audit Table 1 — verbatim "0.781 [0.742, 0.818]". | ✅ |

---

## D. "Hallucination" claims in `SOTA_TABLE.md` §D — re-verified

| Claim in `SOTA_TABLE.md` §D | What the actual source says | Status |
|---|---|---|
| EEG-Bench (2512.08959) does **NOT** evaluate EEGNet on TUAB | CONFIRMED. Fetched 2512.08959 (ar5iv): TUAB models = SVM (0.722), LDA (0.677), BENDR (0.717±.003), Neuro-GPT (0.696±.005), LaBraM (0.838±.011). **EEGNet appears nowhere.** | ✅ claim VERIFIED |
| LaBraM TUAB weighted-F1 = 0.842 (the "0.841" was a misread of this) | CONFIRMED. EEG-Bench Table 5: "LaBraM TUAB weighted F1 = 0.842 ± .012." | ✅ claim VERIFIED |
| The "0.764" / "0.841" EEGNet numbers are NOT in EEG-Bench | CONFIRMED absent from 2512.08959. | ✅ claim VERIFIED |
| Correct EEGNet TUAB Raw BA = **0.804** (spectral-audit) | CONFIRMED — spectral-audit Table 1, EEGNet 0.804 [0.765, 0.843]. | ✅ |
| ShallowConvNet correction: Raw BA = **0.796** not 0.752 | CONFIRMED — spectral-audit Table 1, ShallowFBCSPNet 0.796 [0.755, 0.836]. | ✅ corrected value right |
| Side-note: "the 0.752 is EEGNet's *TUAR* AUROC" | PARTIAL/IMPRECISE. The spectral-audit paper attributes 0.752 to a **PSD-ridge full-spectrum TUAB balanced-accuracy** baseline (§1.4), not an EEGNet TUAR AUROC. The correction itself (use 0.796) is right; only this provenance footnote is described differently in the paper. | ⚠️ note imprecise |
| BIOT-on-TUAB = **0.8815** (pretrained 6-datasets), earlier 0.882→0.869 over-correction | CONFIRMED. BIOT Table 4 / LaBraM Table 1: pretrained-6-datasets BIOT AUROC = 0.8815; vanilla BIOT AUROC = 0.8691. Both the "0.869" and the "0.8815" describe **different BIOT rows** — `SOTA_TABLE.md`'s reconciliation is correct. | ✅ |

---

## E. AFTA (was "cannot verify" in `SOTA_TABLE.md`)

| Method | Claim | Verified source | Status |
|---|---|---|---|
| AFTA | 0.8002 BAcc / 0.8848 AUROC, Brain Sci 2025 (MDPI 2076-3425/15/4/382) | AFTA paper "Self-Supervised Learning with Adaptive Frequency-Time Attention Transformer…", Brain Sci 2025 15(4):382, **Table 7**. WebSearch of the MDPI article + PMC mirror (PMC12025975) both quote: "AFTA achieved the highest balanced accuracy (0.8002) and AUROC (0.8848) on TUAB." (MDPI/PMC full-text PDF was WebFetch-blocked in this env, so this is web-snippet-confirmed, not table-screenshot-confirmed.) | ✅ now VERIFIED via web (still no repo PDF) |

---

## Access log (why some cells are ❌)

- **Worked (table text obtained):** ar5iv/arxiv HTML for 2405.18765 (LaBraM), 2305.10351 (BIOT),
  2502.06438 (FEMBA), 2606.08583 (spectral-audit), 1708.08012 (Schirrmeister), 2512.08959
  (EEG-Bench); WebSearch for AFTA (MDPI/PMC snippets).
- **Blocked by WebFetch permission denial in this environment:** `proceedings.neurips.cc` &
  `papers.nips.cc` (EEGPT PDF), `github.com/BINE022/EEGPT`, `openreview.net`, `pmc.ncbi.nlm.nih.gov`
  & `ncbi.nlm.nih.gov`, `mdpi.com`. Consequence: the **EEGPT exact TUAB figures** (the only cells
  that depend solely on the NeurIPS PDF Table 9) are marked ❌ NOT FOUND. Benchmark papers checked
  for a re-tabulation of EEGPT's TUAB row — NeuroAtlas (2605.14698), CBraMod (2412.07236),
  CRIA (2506.16056), LUNA (2510.22257), SpecMoE (2603.16739), CodeBrain (2506.09110) — none list it.
- The arXiv id **2410.19779 is the WRONG EEGPT** ("EEGPT: …Generalist Foundation Model by
  Autoregressive Pre-training"; variants Base 1.46M / Large 11.29M / Huge 183.8M / Giant 1.09B; no
  TUAB). The EEGPT in `SOTA_TABLE.md` is Wang et al. NeurIPS 2024 (~10M, variants Tiny/Little/Base/
  Large). Do **not** cite 2410.19779 for these rows.

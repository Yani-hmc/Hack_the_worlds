# TUAB SOTA — verified state-of-the-art table

Single authoritative reference for the TUAB normal-vs-abnormal benchmark. Every cell below
was checked against the actual PDF in the repo (file name listed in the **Source** column);
✅ = the value appears verbatim in the cited paper's table.

This file supersedes the older literature tables in `RESULTS_COMPILED.md` and
`LITERATURE_VS_OURS.md` — those still contain a banner pointing here.

---

## Scoring protocol

All numbers below are **per-sample (per-window) Balanced Accuracy / AUROC / AUC-PR** on the
TUAB v3.0.1 evaluation split (409,455 10-second samples from 2,339 recordings, patient-disjoint).
This is the convention BIOT / LaBraM / FEMBA / EEGPT all use — it's the only fair like-for-like
comparison level. Our own per-recording (mean-pool of 16 windows) numbers are clinically
meaningful but ~5 pp higher and are NOT comparable to the values here; see `RESULTS_COMPILED.md`
section B2 if you need those.

---

## A. Verified state of the art (sorted by Balanced Accuracy)

| # | Method | Params | BAcc | AUROC | AUC-PR | Source (PDF in repo) |
|---|---|---|---|---|---|---|
| 1 | LaBraM-Huge | 369M | 0.8258 | 0.9162 | 0.9204 | LaBraM Tbl 1 — `2405.18765v1.pdf` ✅ |
| 2 | LaBraM-Large | 46M | 0.8226 | 0.9127 | 0.9130 | LaBraM Tbl 1 — `2405.18765v1.pdf` ✅ |
| 3 | FEMBA-Huge | 386M | 0.8182 | 0.8921 | 0.9005 | FEMBA Tbl II — `2502.06438v2.pdf` ✅ |
| 4 | FEMBA-Large | 77.8M | 0.8147 | 0.8856 | 0.8992 | FEMBA Tbl II — `2502.06438v2.pdf` ✅ |
| 5 | LaBraM-Base | 5.8M | 0.8140 | 0.9022 | 0.8965 | LaBraM Tbl 1 — `2405.18765v1.pdf` ✅ |
| 6 | FEMBA-Base | 47.7M | 0.8105 | 0.8829 | 0.8894 | FEMBA Tbl II — `2502.06438v2.pdf` ✅ |
| 7 | EEG2Rep | n/a | 0.8052 | 0.8843 | — | FEMBA Tbl II (re-run) — `2502.06438v2.pdf` ✅ |
| 8 | BIOT (pretrained PREST+SHHS) | 3.2M | 0.8019 | 0.8739 | 0.8749 | BIOT Tbl 4 — `2305.10351v1.pdf` ✅ |
| 9 | AFTA | n/a | 0.8002 | 0.8848 | — | Brain Sci 2025 (MDPI 2076-3425/15/4/382) ⚠️ **NO PDF in repo — UNVERIFIED** |
| 10 | EEGPT-25M | 25M | 0.7983 | 0.8718 | — | EEGPT Tbl 2 — `suggestions/papers/eegpt.pdf` ✅ |
| 11 | ST-Transformer | 3.5M | 0.7966 | 0.8707 | 0.8521 | LaBraM Tbl 1 / BIOT Tbl 4 — `2405.18765v1.pdf` ✅ |
| 12 | EEGPT-Tiny | 4.7M | 0.7959 | 0.8716 | — | EEGPT Tbl 2 — `suggestions/papers/eegpt.pdf` ✅ |
| 12 | BIOT (pretrained, 6 EEG datasets) | 3.2M | 0.7959 | 0.8815 | 0.8792 | BIOT Tbl 4; LaBraM Tbl 1 cites this row as "BIOT" — `2305.10351v1.pdf` / `2405.18765v1.pdf` ✅ |
| 14 | BIOT (vanilla, supervised) | 3.2M | 0.7925 | 0.8691 | 0.8707 | BIOT Tbl 4 — `2305.10351v1.pdf` ✅ |
| 15 | SPaRCNet | 0.79M | 0.7896 | 0.8676 | 0.8414 | LaBraM Tbl 1 — `2405.18765v1.pdf` ✅ |
| 16 | FFCL | 2.4M | 0.7848 | 0.8569 | 0.8448 | LaBraM Tbl 1 — `2405.18765v1.pdf` ✅ |
| 17 | CNN-Transformer | 3.2M | 0.7777 | 0.8461 | 0.8433 | LaBraM Tbl 1 — `2405.18765v1.pdf` ✅ |
| 18 | ContraWR | 1.6M | 0.7746 | 0.8456 | 0.8421 | LaBraM Tbl 1 — `2405.18765v1.pdf` ✅ |
| 19 | BENDR (re-run by FEMBA) | 0.39M | 0.7696 | 0.8397 | — | FEMBA Tbl II — `2502.06438v2.pdf` (BENDR's own paper does not report TUAB) ✅ |
| — | BrainBERT | 43.2M | — | 0.8530 | 0.8460 | FEMBA Tbl II — `2502.06438v2.pdf` (BAcc not reported) ✅ |
| — | EEGFormer-Large | 3.2M | — | 0.8760 | 0.8720 | FEMBA Tbl II — `2502.06438v2.pdf` (BAcc not reported) ✅ |
| — | EEGFormer-Base | 2.3M | — | 0.8670 | 0.8670 | FEMBA Tbl II — `2502.06438v2.pdf` (BAcc not reported) ✅ |
| — | EEGFormer-Small | 1.9M | — | 0.8620 | 0.8620 | FEMBA Tbl II — `2502.06438v2.pdf` (BAcc not reported) ✅ |

---

## B. Pre-foundation-model CNN baselines (Schirrmeister 2017 — accuracy-only)

Schirrmeister 2017's TUAB workshop paper reports **accuracy**, not BAcc. BAcc can be derived
from the published sensitivity + specificity.

| Method | Acc | BAcc (derived) | AUROC | Source |
|---|---|---|---|---|
| Deep ConvNet (Deep4Net) | 0.854 | 0.846  *(= (0.751 + 0.941) / 2)* | not reported | Schirrmeister 2017 Tbl II — `1708.08012v3.pdf` ✅ |
| Shallow ConvNet (ShallowFBCSPNet) | 0.845 | 0.839  *(= (0.773 + 0.905) / 2)* | not reported | Schirrmeister 2017 Tbl II — `1708.08012v3.pdf` ✅ |

EEGNet's origin paper (Lawhern 2018, `1611.08024v4.pdf`) does **NOT** evaluate on TUAB — only on
BCI paradigms (P300, ERN, MRCP, motor imagery). Any "EEGNet on TUAB" number is a third-party
re-implementation; see section C.

---

## C. The same simple-CNN architectures re-evaluated by spectral-audit (`2606.08583v2.pdf`)

Different preprocessing protocol → numbers differ by 1-3 pp from the Schirrmeister values above.

| Method | Raw BA [95 % CI] | Source |
|---|---|---|
| EEGNet | 0.804 [0.765, 0.843] | spectral-audit Tbl 1 — `2606.08583v2.pdf` ✅ |
| ShallowFBCSPNet | 0.796 [0.755, 0.836] | spectral-audit Tbl 1 — `2606.08583v2.pdf` ✅ |
| Deep4Net | 0.816 [0.777, 0.854] | spectral-audit Tbl 1 — `2606.08583v2.pdf` ✅ |
| BIOT | 0.802 [0.781, 0.823] | spectral-audit Tbl 1 — `2606.08583v2.pdf` ✅ |
| LaBraM | 0.786 [0.763, 0.809] | spectral-audit Tbl 1 — `2606.08583v2.pdf` ✅ |
| EEGPT | 0.801 [0.779, 0.824] | spectral-audit Tbl 1 — `2606.08583v2.pdf` ✅ |
| CBraMod | 0.770 [0.746, 0.795] | spectral-audit Tbl 1 — `2606.08583v2.pdf` ✅ |
| REVE | 0.780 [0.753, 0.806] | spectral-audit Tbl 1 — `2606.08583v2.pdf` ✅ |
| EEGMamba | 0.781 [0.756, 0.807] | spectral-audit Tbl 1 — `2606.08583v2.pdf` ✅ |
| BENDR † | 0.781 [0.742, 0.818] | spectral-audit Tbl 1 — `2606.08583v2.pdf` ✅ |

---

## D. ❌ Hallucinations in earlier drafts (now corrected here)

Two cells previously in `RESULTS_COMPILED.md` / `LITERATURE_VS_OURS.md` were **wrong** — the
cited PDF does not contain those numbers. Both were authored by an offline agent without paper
access; this section documents the corrections so the history is auditable.

| Method | Earlier (wrong) claim | "Cited" source | What the source actually says | Correct value to use |
|---|---|---|---|---|
| EEGNet on TUAB | 0.764 BAcc / 0.841 AUROC, "EEG-Bench arXiv:2512.08959" | EEG-Bench (`2512.08959v1.pdf`) | **EEG-Bench does NOT evaluate EEGNet at all** — its Tables 3 and 5 contain only SVM / LDA / BENDR / Neuro-GPT / LaBraM. The "0.841" appears to be a misread of LaBraM's TUAB **weighted F1 = 0.842** (Tbl 5) — wrong method. The "0.764" is **not in the paper**. | spectral-audit Raw BA `0.804` (`2606.08583v2.pdf`), OR explicitly state "no primary-paper number — Lawhern 2018 does not evaluate TUAB". |
| ShallowConvNet on TUAB | ~0.752 BAcc, "spectral-audit arXiv:2606.08583" | spectral-audit (`2606.08583v2.pdf`) | spectral-audit Tbl 1 reports **ShallowFBCSPNet Raw BA = 0.796**, not 0.752. The 0.752 in spectral-audit is **EEGNet's *TUAR* AUROC** (different method, dataset, and metric — triple confusion). | spectral-audit Raw BA `0.796`, OR Schirrmeister-derived BAcc `0.839` (sens 0.773 + spec 0.905) / 2. |

A third earlier figure (BIOT AUROC) was hallucinated in an even earlier draft and has already
been corrected in `LITERATURE_VS_OURS.md`:

| Method | Earlier wrong claim → over-correction | Truth |
|---|---|---|
| BIOT on TUAB | 0.882 → over-corrected to 0.869 | 0.8815 (BIOT pretrained on 6 EEG datasets, BIOT Tbl 4 row 11 / LaBraM Tbl 1) ✅ |

---

## E. ⚠️ Cannot verify

| Method | Claim | Reason | Action |
|---|---|---|---|
| AFTA | 0.8002 BAcc / 0.8848 AUROC, Brain Sci 2025 (MDPI 2076-3425/15/4/382) | **No PDF in the repo** | Upload the Brain Sci 2025 paper to `suggestions/papers/`, OR drop the row, OR keep it tagged ⚠️. |

---

## F. Footnotes to attach to this table when publishing

1. **Scoring.** All numbers are per-sample (per-window) Balanced Accuracy / AUROC / AUC-PR on
   TUAB v3.0.1 — the standard BIOT / LaBraM / FEMBA / EEGPT convention. Our per-recording
   numbers in `RESULTS_COMPILED.md` section B2 are NOT comparable (~5 pp higher due to
   mean-pool aggregation).
2. **Origin-paper vs benchmark-paper.** For methods whose origin paper does not evaluate on
   TUAB (EEGNet, ShallowConvNet, Deep4Net, BENDR, BrainBERT), we cite the most-canonical
   benchmark paper that re-runs them. Same-method numbers can differ 1-3 pp between
   re-implementations due to preprocessing / split / seed choices.
3. **AFTA** is currently unverified — no source PDF in the repository.
4. **Hallucination policy.** Two cells in the earlier team drafts were wrong (section D).
   Going forward: **every literature cell in a results table must cite a PDF that physically
   exists in this repo**, and every reviewer should be able to grep the cited PDF for the
   exact number.

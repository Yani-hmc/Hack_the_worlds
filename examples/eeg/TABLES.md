# Result tables — papers vs our experiments (four metrics)

**Read first:** the TUAB/TUEV papers report only **Balanced Accuracy / AUC-PR / AUROC** (TUAB) and
**Balanced Accuracy / Cohen-κ / Weighted-F1** (TUEV). They do **NOT** report accuracy/precision/
recall/F1. So those four metrics exist **only for our runs**; the fair cross-comparison columns are
**Balanced Accuracy** and **AUROC**. Our per-window rows use the papers' protocol; per-recording is
a clinical extra and is **not** comparable to the papers.

---

## TABLE 1 — Papers, TUAB (source: LaBraM ICLR 2024, Table 1)
| Method | Balanced Acc | AUC-PR | AUROC |
|---|---|---|---|
| SPaRCNet | 0.7896 | 0.8414 | 0.8676 |
| ContraWR | 0.7746 | 0.8421 | 0.8456 |
| CNN-Transformer | 0.7777 | 0.8433 | 0.8461 |
| FFCL | 0.7848 | 0.8448 | 0.8569 |
| ST-Transformer | 0.7966 | 0.8521 | 0.8707 |
| BIOT | 0.7959 | 0.8792 | 0.8815 |
| LaBraM-Base | 0.8140 | 0.8965 | 0.9022 |
| LaBraM-Large | 0.8226 | 0.9130 | 0.9127 |
| LaBraM-Huge | 0.8258 | 0.9204 | 0.9162 |

## TABLE 2 — Our experiments, TUAB, PER-WINDOW (papers' protocol → comparable)
| Our model | Accuracy | Precision | Recall | F1 | Balanced Acc | AUROC |
|---|---|---|---|---|---|---|
| JEPA base (VICReg) | 0.759 | 0.744 | 0.721 | 0.732 | 0.756 | 0.845 |
| JEPA + corruption (VICReg) | 0.774 | 0.765 | 0.729 | 0.746 | 0.770 | 0.848 |
| **JEPA + corruption (SIGReg)** | 0.778 | 0.766 | 0.738 | 0.752 | **0.775** | **0.856** |
| JEPA + spectral | 0.768 | 0.754 | 0.730 | 0.741 | 0.765 | 0.849 |
| JEPA multi-corpus (13k recs) | 0.778 | 0.774 | 0.725 | 0.749 | 0.774 | 0.854 |
| EEGNet (ours, supervised) | 0.802 | 0.818 | 0.729 | 0.771 | 0.796 | 0.882 |
| ShallowConvNet (ours, supervised) | 0.776 | 0.742 | 0.782 | 0.762 | 0.777 | 0.857 |

## TABLE 3 — Our experiments, TUAB, PER-RECORDING (clinical; NOT comparable to papers)
| Our model | Accuracy | Precision | Recall | F1 | Balanced Acc | AUROC |
|---|---|---|---|---|---|---|
| JEPA base (VICReg) | 0.801 | 0.803 | 0.746 | 0.774 | 0.796 | 0.888 |
| JEPA + corruption (VICReg) | 0.830 | 0.844 | 0.770 | 0.805 | 0.825 | 0.904 |
| JEPA + corruption (SIGReg) | 0.830 | 0.844 | 0.770 | 0.805 | 0.825 | 0.913 |
| JEPA + spectral | 0.841 | 0.853 | 0.786 | 0.818 | 0.836 | 0.887 |
| JEPA multi-corpus (frozen) | 0.819 | 0.846 | 0.738 | 0.788 | 0.812 | 0.883 |
| **JEPA fine-tune (TUAB init)** | 0.844 | 0.888 | 0.754 | 0.816 | **0.837** | **0.919** |
| JEPA fine-tune (multi-corpus init) | 0.844 | 0.888 | 0.754 | 0.816 | 0.837 | 0.918 |
| EEGNet (ours) | 0.830 | 0.856 | 0.754 | 0.802 | 0.824 | 0.913 |
| ShallowConvNet (ours) | 0.804 | 0.786 | 0.786 | 0.786 | 0.803 | 0.893 |

---

## TABLE 4 — Papers, TUEV 6-class (source: LaBraM ICLR 2024, Table 2)
| Method | Balanced Acc | Cohen-κ | Weighted-F1 |
|---|---|---|---|
| SPaRCNet | 0.4161 | 0.4233 | 0.7024 |
| ContraWR | 0.4384 | 0.3912 | 0.6893 |
| CNN-Transformer | 0.4087 | 0.3815 | 0.6854 |
| FFCL | 0.3979 | 0.3732 | 0.6783 |
| ST-Transformer | 0.3984 | 0.3765 | 0.6823 |
| BIOT | 0.5281 | 0.5273 | 0.7492 |
| LaBraM-Base | 0.6409 | 0.6637 | 0.8312 |
| LaBraM-Large | 0.6581 | 0.6622 | 0.8315 |
| LaBraM-Huge | 0.6616 | 0.6745 | 0.8329 |

## TABLE 5 — Our experiment, TUEV 6-class (frozen TUAB encoder, transfer)
| Model | Accuracy | Balanced Acc | Macro-F1 | Weighted-F1 | Cohen-κ |
|---|---|---|---|---|---|
| JEPA SIGReg (frozen, transfer) | 0.405 | 0.364 | 0.312 | 0.431 | 0.197 |
| Random encoder (floor) | 0.355 | 0.337 | 0.278 | 0.401 | 0.164 |

(Ours is a frozen TUAB-encoder transfer with a different windowing protocol → below the trained
models, above the random floor.)

---

## One-line reading
Per-window, our best JEPA = **0.775 BAcc / 0.856 AUROC**: ≈ ContraWR/FFCL, below SPaRCNet/
ST-Transformer/BIOT, well below LaBraM. We are a clean, honest baseline — not SOTA. The gap is a
model-capacity gap (small conv encoder vs big transformer), not a data gap (multi-corpus didn't help).

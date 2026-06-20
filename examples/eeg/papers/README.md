# Source papers for the TUAB SOTA table

Every literature number in [`../SOTA_TABLE.md`](../SOTA_TABLE.md) is backed by one of the PDFs
in this folder — so any reviewer can open the file and grep the exact value. Downloaded from
the canonical open-access source (arXiv / OpenReview / EuropePMC).

| PDF | Paper | Venue | TUAB rows it backs |
|---|---|---|---|
| `LaBraM_2405.18765.pdf` | LaBraM — Large Brain Model (Jiang et al.) | ICLR 2024 | LaBraM-Base/Large/Huge (Tbl 1) + the 6 baselines SPaRCNet/ContraWR/CNN-T/FFCL/ST-T/BIOT it re-runs |
| `BIOT_2305.10351.pdf` | BIOT — Biosignal Transformer (Yang et al.) | NeurIPS 2023 | BIOT vanilla/PREST+SHHS/6-datasets, AUC-PR for the baselines (Tbl 4) |
| `FEMBA_2502.06438.pdf` | FEMBA — Mamba EEG foundation model (Tegon et al.) | 2025 | FEMBA-Base/Large/Huge, EEG2Rep, BENDR, BrainBERT, EEGFormer (Tbl II) |
| `EEGPT_NeurIPS2024.pdf` | EEGPT — Pretrained Transformer (Wang et al.) | NeurIPS 2024 | EEGPT-Tiny (4.7M) & EEGPT-25M (Tbl 3 main / Tbl 9 appendix) |
| `AFTA_brainsci-15-00382.pdf` | AFTA — Adaptive Frequency-Time Attention (Huang et al.) | Brain Sci 2025 15(4):382 | AFTA (Tbl 3) |
| `Schirrmeister_1708.08012.pdf` | Deep/Shallow ConvNet (Schirrmeister et al.) | HBM 2017 | Deep4Net, ShallowConvNet (Tbl II, accuracy) |
| `SpectralAudit_2606.08583.pdf` | Spectral-audit of EEG benchmarks | 2026 | Raw-BA re-eval of EEGNet/Shallow/Deep4/BIOT/LaBraM/EEGPT/CBraMod/REVE/EEGMamba/BENDR (Tbl 1) |
| `EEGBench_2512.08959.pdf` | EEG-Bench | 2025 | cross-check that EEGNet is NOT evaluated on TUAB; LaBraM weighted-F1 0.842 |

Note: the EEGPT and AFTA PDFs were the two papers needed to verify the previously-unconfirmed
EEGPT (0.7983/0.8718, 0.7959/0.8716) and AFTA (0.8002/0.8848) TUAB rows. Both now confirmed.

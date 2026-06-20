# TUAB per-recording comparison — all models we ran ourselves (vote/mean-pool aggregation)

**What this is.** Every model below was run *by us* on the **same** TUAB_PREPROCESSED data
(19ch, 200Hz, 10s windows, patient-disjoint eval), and scored at the **recording level** by
mean-pooling each recording's 16 window-probabilities into one prediction (the "vote trick").
Because all rows share identical data + identical aggregation, this is a clean apples-to-apples
panel — including the SOTA models, which we run from their **official repos** rather than cite.

> ⚠️ **Do not cross-compare with the per-window literature table.** Per-recording scores run
> ~5 pp higher than per-window (pooling 16 windows cancels noise). The published SOTA numbers
> (LaBraM 0.814, BIOT 0.7959, …) are **per-window** — they are NOT in this table and a 0.83 here
> does not "beat" a 0.81 there. The two levels are in different files on purpose:
> per-window/literature → [`SOTA_TABLE_VERIFIED.md`](SOTA_TABLE_VERIFIED.md); per-recording (this file).

## Per-recording — four headline scores (sorted by Balanced Accuracy)

| Model | Provenance | Acc | **BAcc** | F1 | **AUROC** |
|---|---|---|---|---|---|
| EB-JEPA fine-tune (corruption init) | **ours** (JEPA) | 0.844 | **0.837** | 0.816 | **0.919** |
| EB-JEPA + spectral 0.1 (VICReg, frozen probe) | **ours** (JEPA) | 0.841 | 0.836 | 0.818 | 0.887 |
| BIOT — vanilla (supervised) | **official `ycq091044/BIOT`**, our data | 0.837 | 0.829 | 0.805 | 0.903 |
| EB-JEPA + corruption (VICReg, frozen probe) | **ours** (JEPA) | 0.830 | 0.825 | 0.805 | 0.904 |
| EB-JEPA + corruption (SIGReg, frozen probe) | **ours** (JEPA) | — | 0.825 | 0.805 | 0.913 |
| BIOT — pretrained (6 EEG datasets ckpt) | **official `ycq091044/BIOT`**, our data | 0.833 | 0.824 | 0.798 | 0.882 |
| EEGNet | **ours** (braindecode EEGNetv4) | 0.830 | 0.824 | 0.802 | 0.913 |
| ShallowConvNet | **ours** (braindecode) | 0.804 | 0.803 | 0.786 | 0.893 |
| JEPA-Transformer + multi-corpus (13k rec) | **ours** (ViT, MLP probe) | 0.804 | 0.798 | 0.773 | 0.872 |
| SPaRCNet | **official BIOT repo**, our data | _running (job 75826)_ | | | |
| ContraWR | **official BIOT repo**, our data | _running_ | | | |
| CNN-Transformer | **official BIOT repo**, our data | _running_ | | | |
| FFCL | **official BIOT repo**, our data | _running_ | | | |
| ST-Transformer | **official BIOT repo**, our data | _running_ | | | |
| **LaBraM-Base (fine-tune)** | **official `935963004/LaBraM`** + pretrained ckpt | _running (job 75864)_ | | | |

Precision / Recall / AUC-PR are logged too (new runs add AUC-PR); omitted here for the 4-score view.

## Honest reading

- Single-seed rows vary ±0.005. The corruption-JEPA **seed-averaged** (3 seeds) per-recording is
  **BAcc 0.819 ± 0.004 / AUROC 0.900 ± 0.006** — so the 0.825–0.837 frozen/fine-tune rows are all
  within noise of ~0.82. Don't over-read a single 0.837.
- At the recording level our JEPA is **on par with BIOT-from-source** (0.825–0.837 vs 0.824–0.829
  BAcc) and **on par with EEGNet** (0.824). The real test is the pending **LaBraM** row: if LaBraM
  fine-tuned from its pretrained checkpoint lands clearly above ~0.84 here, that's the genuine SOTA
  gap (its massive pretraining corpus), measured on *our* recordings.
- BIOT pretrained ≈ BIOT vanilla here (0.824 vs 0.829) because we fine-tune on our data for only
  20 epochs with our preprocessing — the pretraining edge mostly shows at per-window on their own
  pipeline, not after our short recording-level fine-tune.

## How each "from source" number is produced

- **BIOT / SPaRCNet / ContraWR / CNN-Transformer / FFCL / ST-Transformer** — the exact model classes
  from `github.com/ycq091044/BIOT`'s `model.py`, constructed with the official
  `run_binary_supervised.py` TUAB hyper-params, trained 20 epochs on our windows, eval mean-pooled
  to recordings. Code: [`baselines/biot/adapter_baselines.py`](../../baselines/biot/adapter_baselines.py).
- **LaBraM-Base** — the official `NeuralTransformer` from `github.com/935963004/LaBraM`
  `modeling_finetune.py`, loaded from the repo's pretrained `labram-base.pth` (student encoder),
  fine-tuned end-to-end on our windows. Two documented deviations (input is our z-score, not µV/100;
  canonical TUAB channel order assumed). Code:
  [`baselines/labram/adapter_labram.py`](../../baselines/labram/adapter_labram.py).

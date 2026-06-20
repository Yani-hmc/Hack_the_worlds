# TUAB per-recording comparison — all models we ran ourselves (vote/mean-pool aggregation)

**What this is.** Every model below was run *by us* on the **same** TUAB_PREPROCESSED data
(19ch, 200Hz, 10s windows, patient-disjoint eval), and scored at the **recording level** by
mean-pooling each recording's 16 window-probabilities into one prediction (the "vote trick").
Because all rows share identical data + identical aggregation, this is a clean apples-to-apples
panel — including the SOTA models, which we run from their **official repos** rather than cite.

> ⚠️ **Do not cross-compare with the per-window literature table.** Per-recording scores run
> ~5 pp higher than per-window. The published SOTA numbers (LaBraM 0.814, BIOT 0.7959, …) are
> **per-window** — they are NOT in this table; per-window/literature → [`SOTA_TABLE.md`](SOTA_TABLE.md).
>
> ⚠️ **Single-run vs seed-averaged.** Most rows are a single run (±0.005–0.017 seed noise). The two
> rows that drive the "SSL vs supervised" comparison — **EB-JEPA fine-tune** and **EEGNet** — are
> reported **3-seed, final-epoch** (see the fair head-to-head below). An earlier best-epoch reading
> of the JEPA fine-tune (0.837/0.919) was **test-set peeking and is retracted** → 0.812/0.908.

## Per-recording — four headline scores (sorted by Balanced Accuracy)

| Model | Provenance | Acc | **BAcc** | F1 | **AUROC** |
|---|---|---|---|---|---|
| **LaBraM-Base (fine-tune)** | **official `935963004/LaBraM`** + pretrained ckpt | 0.851 | **0.846** | 0.829 | **0.926** |
| EB-JEPA + spectral 0.1 (frozen, 1 seed) | **ours** (JEPA) | 0.841 | 0.836 | 0.818 | 0.887 |
| FFCL | **official BIOT repo**, our data | 0.833 | 0.833 | 0.820 | 0.908 |
| ContraWR | **official BIOT repo**, our data | 0.837 | 0.830 | 0.807 | 0.912 |
| BIOT — vanilla (supervised) | **official `ycq091044/BIOT`**, our data | 0.837 | 0.829 | 0.805 | 0.903 |
| EB-JEPA + corruption (VICReg frozen, 1 seed) | **ours** (JEPA) | 0.830 | 0.825 | 0.805 | 0.904 |
| EB-JEPA + corruption (SIGReg frozen, 1 seed) | **ours** (JEPA) | — | 0.825 | 0.805 | 0.913 |
| BIOT — pretrained (6-dataset ckpt) | **official `ycq091044/BIOT`**, our data | 0.833 | 0.824 | 0.798 | 0.882 |
| EEGNet (3-seed, final epoch) | **ours** (braindecode) | — | **0.812** ± .013 | — | 0.911 ± .007 |
| **EB-JEPA fine-tune (3-seed, final epoch)** | **ours** (JEPA) | 0.820 | **0.812** ± .004 | 0.786 | 0.908 ± .006 |
| ShallowConvNet (1 seed) | **ours** (braindecode) | 0.804 | 0.803 | 0.786 | 0.893 |
| CNN-Transformer | **official BIOT repo**, our data | 0.804 | 0.802 | 0.784 | 0.885 |
| SPaRCNet | **official BIOT repo**, our data | 0.801 | 0.800 | 0.783 | 0.905 |
| JEPA-Transformer + multi-corpus (13k rec) | **ours** (ViT, 1 seed) | 0.804 | 0.798 | 0.773 | 0.872 |
| ST-Transformer | **official BIOT repo**, our data | _re-running (job 76129; timed out first pass)_ | | | |

EEGNet's single-seed run was 0.824 BAcc / 0.913 AUROC; its 3-seed final-epoch mean is **0.812 ± .013**
(used above for the fair comparison). LaBraM per-window = 0.806 BAcc / 0.855 AUROC / 0.913 AUC-PR.

## Fair head-to-head (3 seeds, final epoch, no peeking) — the defensible SSL-vs-supervised claim

| Model | BAcc | AUROC |
|---|---|---|
| EB-JEPA fine-tuned (**no labels in pretraining**) | 0.812 ± 0.004 | 0.908 ± 0.006 |
| EEGNet (supervised) | 0.812 ± 0.013 | 0.911 ± 0.007 |

→ **A statistical tie.** Self-supervised JEPA **matches** a properly-trained supervised EEGNet without
using any labels during pretraining. (The retracted single-seed best-epoch reading of 0.837 vs 0.824
made SSL look like it *beat* EEGNet; selecting the best epoch on the eval set is test-set peeking. Both
honestly average to 0.812 — see `reports/BUGS_AND_MISTAKES.md` Bug 0.)

## Honest reading

- **LaBraM-Base leads the table at 0.846 / 0.926** — ~3 pp above our JEPA fine-tune (0.812, 3-seed) and
  ~1.7 pp above BIOT-from-source (0.829). A true foundation model opens a real gap on *our exact
  recordings* — the genuine SOTA pretraining advantage, not a benchmark-pipeline artefact.
- **Below LaBraM the field is bunched at 0.80–0.83** (BIOT, FFCL, ContraWR, our frozen JEPA, EEGNet,
  our fine-tuned JEPA). Our self-supervised JEPA sits squarely in that band — matching the supervised
  CNN/transformer baselines, beaten only by the large pretrained foundation model.
- **LaBraM run is faithful:** its per-window BAcc 0.806 is within ~1 pp of the published 0.814 despite
  our two documented deviations (z-scored input, 20-epoch fine-tune) — the adapter reproduces the real
  model. The official BIOT-repo baselines also land near their published per-window values once
  aggregated.

## How each "from source" number is produced

- **BIOT / SPaRCNet / ContraWR / CNN-Transformer / FFCL / ST-Transformer** — the exact model classes
  from `github.com/ycq091044/BIOT`'s `model.py`, constructed with the official
  `run_binary_supervised.py` TUAB hyper-params, trained 20 epochs on our windows, eval mean-pooled
  to recordings. Code: [`baselines/biot/adapter_baselines.py`](../../baselines/biot/adapter_baselines.py).
- **LaBraM-Base** — the official `NeuralTransformer` from `github.com/935963004/LaBraM`
  `modeling_finetune.py`, loaded from the repo's pretrained `labram-base.pth` (student encoder),
  fine-tuned end-to-end on our windows. Code:
  [`baselines/labram/adapter_labram.py`](../../baselines/labram/adapter_labram.py).

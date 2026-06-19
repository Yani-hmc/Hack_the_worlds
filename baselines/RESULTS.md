# Phase 0 — TUAB Baseline Results

Binary task: **normal (0) vs abnormal (1)**, patient-disjoint `train`/`eval`.
See `PHASE0_BASELINES.md` for the literature review, sourcing, and full method notes.

Two groups:
- **Group A — Paper-reported on TUAB** (pre-filled from the literature; `⟨verify⟩` = recorded
  offline, confirm against cited table). Papers report only **Balanced-Acc / AUROC**
  (BIOT also AUC-PR); **precision/recall/F1 are not reported on TUAB** → left blank.
- **Group B — Ours on TUAB** (RUN ON DALIA — TODO). Our adapters print **all six** metrics,
  at **per-recording** level (apples-to-apples with the JEPA frozen probe) and per-window.

> ⚠️ **METRIC MISMATCH:** compare like-for-like on **Balanced-Acc** and **AUROC** only.
> Precision/recall/F1 exist only for our runs.
> ⚠️ **MONTAGE:** literature uses **16 ch**; ours uses **19 ch**. Minor, defensible deviation.
> ⚠️ **SCORING:** literature scores **per-window**; our headline row is **per-recording**
>   (mean window-prob). Each adapter prints both so you can match either convention.

---

## Group A — Paper-reported on TUAB (literature; `⟨verify⟩`)

| Method | accuracy | balanced-acc | precision | recall | F1 | AUROC | Source |
|---|---|---|---|---|---|---|---|
| EEGNet         | — | ~0.745 ⟨v⟩ | — | — | — | ~0.835 ⟨v⟩ | BIOT/LaBraM tables |
| ShallowConvNet | — | ~0.845 ⟨v⟩ | — | — | — | — (acc only) | Gemein 2020 |
| SPaRCNet       | — | 0.7896 ⟨v⟩ | — | — | — | 0.8676 ⟨v⟩ | BIOT Table 3 |
| ContraWR       | — | 0.7746 ⟨v⟩ | — | — | — | 0.8456 ⟨v⟩ | BIOT Table 3 |
| CNN-Transformer| — | 0.7777 ⟨v⟩ | — | — | — | 0.8461 ⟨v⟩ | BIOT Table 3 |
| FFCL           | — | 0.7848 ⟨v⟩ | — | — | — | 0.8569 ⟨v⟩ | BIOT Table 3 |
| ST-Transformer | — | 0.7966 ⟨v⟩ | — | — | — | 0.8707 ⟨v⟩ | BIOT Table 3 |
| **BIOT**       | — | **0.7959** ⟨v⟩ | — | — | — | **0.8815** ⟨v⟩ | BIOT Table 3 (AUC-PR 0.8792) |
| **LaBraM-Base**| — | **0.8140** ⟨v⟩ | — | — | — | **0.9022** ⟨v⟩ | LaBraM Table 2 (AUC-PR 0.8965) |
| EEGPT          | — | ~0.795 ⟨v⟩ | — | — | — | ~0.87 ⟨v⟩ | EEGPT (NeurIPS'24) ⟨v⟩ |
| BENDR          | — | n/r (re-impl ~0.76) | — | — | — | n/r (~0.86) | not in own paper |
| Brant          | — | n/a (intracranial) | — | — | — | n/a | does not use TUAB |
| EEG-Conformer  | — | not reported on TUAB | — | — | — | not reported | BCI IV-2a acc ~0.78 instead |

---

## Group B — Ours on TUAB (RAN ON DALIA — per-recording, patient-disjoint, n_eval=276)

Actual runs on the DALIA GB200 cluster. See `examples/eeg/RESULTS.md` for the full JEPA
energy-strategy ablation and reproduction commands.

| Method | accuracy | balanced-acc | precision | recall | F1 | AUROC |
|---|---|---|---|---|---|---|
| EEGNet (ours, supervised)        | 0.830 | 0.824 | 0.856 | 0.754 | 0.802 | 0.913 |
| ShallowConvNet (ours, supervised)| 0.804 | 0.803 | 0.786 | 0.786 | 0.786 | 0.893 |
| BIOT (ours, supervised)          | TODO (needs repo clone, see biot/README.md) | | | | | |
| **EB-JEPA (ours, frozen probe, +corruption)** | 0.830 | 0.825 | 0.844 | 0.770 | 0.805 | **0.904** |
| **EB-JEPA (ours, fine-tuned from corruption)** | — | **0.837** | — | — | 0.816 | **0.919** |

> Best overall = fine-tuned EB-JEPA (AUROC 0.919). The **frozen** self-supervised probe
> (0.825/0.904) already matches supervised EEGNet (0.824/0.913) and the BIOT/LaBraM literature.
> vs Group A (balanced-acc / AUROC, like-for-like): BIOT 0.796/0.882, LaBraM 0.814/0.902.

### Exact commands to fill each row (run on a Dalia COMPUTE node)

Always grab a node first:
```bash
srun --partition=defq --reservation=Vivatech --gres=gpu:1 \
     --cpus-per-task=36 --time=02:00:00 --pty bash
```

**EEGNet** — copy the `per-recording` dict it prints:
```bash
uv run python -m baselines.eegnet.train_eegnet_tuab --epochs 30 --batch-size 256 --lr 1e-3
```

**ShallowConvNet**:
```bash
uv run python -m baselines.shallowconvnet.train_shallowconvnet_tuab \
    --epochs 30 --batch-size 256 --lr 6.25e-4
```

**BIOT** (clone repo first per `baselines/biot/README.md`):
```bash
PYTHONPATH=/lustre/work/vivatech-<team>/$USER/BIOT:$PYTHONPATH \
  uv run python baselines/biot/adapter_biot.py --epochs 20 --batch-size 128 --lr 1e-3 --n-channels 19
```

**EB-JEPA** (our model — pretrain then frozen probe; gives the number to beat/compare):
```bash
uv run python -m examples.eeg.main  --fname examples/eeg/cfgs/train.yaml
uv run python -m examples.eeg.eval  --ckpt <ckpt_dir>/latest.pth.tar
# copy the printed dict from examples/eeg/eval.py
```

---

## Headline comparison (fill the `ours` numbers after Dalia runs)

The peers to read our JEPA against, on **Balanced-Acc / AUROC**:
- **Floor:** EEGNet ~0.745 / ~0.835.
- **Strong classical-DL:** ShallowConvNet ~0.845 BAcc (no AUROC).
- **Primary transformer peer (same 200Hz/10s spec):** BIOT 0.7959 / 0.8815.
- **Upper bound:** LaBraM-Base 0.8140 / 0.9022.

A JEPA frozen-probe sitting near/above BIOT (~0.80 / ~0.88) would be a strong Phase-0 result;
near EEGNet (~0.75) is the floor.

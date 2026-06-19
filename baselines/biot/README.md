# BIOT baseline on our TUAB pipeline (clone + adapt)

**BIOT** (Yang, Westover, Sun — *Biosignal Transformer for Cross-data Learning in the Wild*,
NeurIPS 2023, arXiv:2305.10351) is the transformer baseline that reports TUAB at **exactly
our input spec** (200 Hz, 10 s windows). License **MIT** `⟨verify the repo LICENSE⟩`, so it
is safe to vendor. Because BIOT ships **pretrained checkpoints** and a non-trivial
tokenizer/dependency stack, we do NOT reimplement it in-file — instead, clone the official
repo and plug OUR dataset into it. Exact steps below.

Paper-reported TUAB (BIOT Table 3, `⟨verify⟩`): **Balanced-Acc ≈ 0.7959, AUROC ≈ 0.8815,
AUC-PR ≈ 0.8792**.

---

## 0. Input contract (what BIOT expects vs what we have)

| | BIOT expects | We provide | Action |
|---|---|---|---|
| Channels | 16 (its TUAB montage) | **19** | pass 19 (BIOT's channel-token design supports variable C) OR slice to its 16-channel order |
| Sampling | 200 Hz | **200 Hz** | none |
| Window | 10 s (2000 samp) | **10 s (2000 samp)** | none |
| Scaling | BIOT does its own per-sample norm | **already per-channel z-scored** | fine; optionally disable BIOT's extra norm to avoid double-normalizing |

Our `[19, 2000]` z-scored window is essentially drop-in. The only real decision is 19 vs 16
channels — start with **19** (BIOT tokenizes per channel and adds a learned channel
embedding, so extra channels just add tokens), and if you want a strict reproduction of the
paper montage, build the 16-channel index map and slice.

---

## 1. Clone the official repo (do this on the Dalia LOGIN node — compute nodes have no internet)

```bash
cd /lustre/work/vivatech-<team>/$USER
git clone https://github.com/ycq091044/BIOT.git
# optional pretrained weights: download per the repo README into BIOT/pretrained-models/
```

## 2. Install its deps into the project venv (login node)

```bash
cd /lustre/work/vivatech-<team>/$USER/eb_jepa
uv pip install linear-attention-transformer  # BIOT's transformer dep
# torch / numpy / scikit-learn already present from the eb_jepa env
```

## 3. Drop in the adapter

Copy `adapter_biot.py` (in this folder) next to BIOT's source, or run it from the repo root
with `PYTHONPATH` pointing at both the cloned `BIOT/` and our `eb_jepa`. It:
- imports `BIOTClassifier` from the cloned repo,
- imports OUR `EEGDataset` (probe mode) and flattens it to labelled `[19, 2000]` windows,
- trains supervised on TUAB **train** patients, evaluates **patient-disjoint** on **eval**,
- prints accuracy / balanced-acc / precision / recall / F1 / AUROC, per-window AND
  aggregated to recording level (mean window-prob) to match our JEPA frozen probe.

## 4. Run (Dalia COMPUTE node)

```bash
srun --partition=defq --reservation=Vivatech --gres=gpu:1 \
     --cpus-per-task=36 --time=02:00:00 --pty bash

# from the eb_jepa repo root, with the cloned BIOT on PYTHONPATH:
PYTHONPATH=/lustre/work/vivatech-<team>/$USER/BIOT:$PYTHONPATH \
  uv run python baselines/biot/adapter_biot.py \
    --epochs 20 --batch-size 128 --lr 1e-3 --n-channels 19
```

### Strict-reproduction alternative (use BIOT's own TUAB script)
BIOT's repo includes `run_binary_supervised.py` for TUAB. To reproduce the *paper* number
directly, preprocess our TUAB EDFs into BIOT's expected `.pkl` sample format (16 ch, 200 Hz,
10 s) using the repo's `datasets/` preprocessing, then run their script. This bypasses our
loader but reproduces the literature value 1:1. Our `adapter_biot.py` instead keeps OUR
loader/split for an apples-to-apples comparison with the JEPA probe — prefer the adapter for
the comparison, the repo script only if you need to replicate the exact 0.7959/0.8815.

---

## Caveats
- `⟨verify⟩` all paper numbers against **BIOT Table 3**; recorded offline.
- Confirm the repo **LICENSE is MIT** before vendoring.
- 19 vs 16 channels is a minor montage deviation — note it in `RESULTS.md`.
- If you load BIOT's **pretrained** checkpoint and fine-tune, report it as "BIOT (pretrained,
  fine-tuned)"; training from scratch on our split is "BIOT (supervised, ours)".

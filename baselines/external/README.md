# Official baseline repos — clone-and-run status

Per "use the real code if accessible, drop if not" rule. Repos are cloned to
`$WORK/external/<NAME>/` on Dalia by [setup_externals.sh](setup_externals.sh).

| Model | Repo | Status on our TUAB_PREPROCESSED pipeline | Adapter |
|---|---|---|---|
| **BIOT** | [ycq091044/BIOT](https://github.com/ycq091044/BIOT) | ✅ **RUNNABLE** — pretrained ckpts in repo, 16/18-ch variants, 200 Hz / 10 s = matches us. Slice our 19 ch to 18 to use the "6 datasets" pretrained model (the one cited as 0.7959 BAcc in LaBraM Tbl 1). | [baselines/biot/adapter_biot.py](../biot/adapter_biot.py) + [baselines/biot/run_biot.sbatch](../biot/run_biot.sbatch) |
| **EEGPT** | [BINE022/EEGPT](https://github.com/BINE022/EEGPT) | ⚠️ **PARTIAL** — official model + downstream scripts cloned, but pretrained weights are on **figshare** (manual download required, not in repo). Pretraining is on **58 channels / 256 Hz / 4 s windows** (vs our 19 / 200 / 10) — using the pretrained weights would require resampling + channel zero-padding, breaking apples-to-apples. *Not run* until either (a) someone wants to do the cross-protocol comparison or (b) we drop the "fair" requirement. |
| **BENDR** | [SPOClab-ca/BENDR](https://github.com/SPOClab-ca/BENDR) | ⚠️ **PARTIAL** — official code cloned, pretrained weights on GitHub releases, but requires **DN3** library (heavy dep, custom dataset format). BENDR's own paper doesn't evaluate TUAB; the 0.770 / 0.840 we cite for it comes from FEMBA Tbl II's re-run. Re-running BENDR on our pipeline = port DN3 dataset format adapter. *Not run.* |
| **FEMBA (BioFoundation)** | [pulp-bio/BioFoundation](https://github.com/pulp-bio/BioFoundation) | ⚠️ **PARTIAL** — official code cloned, pretrained weights on HuggingFace (`PulpBio/FEMBA`). Uses **Mamba state-space kernels**; the custom CUDA kernels may not build on aarch64 GB200 (untested). *Not run.* |
| **LaBraM** | [935963004/LaBraM](https://github.com/935963004/LaBraM) | ❌ **DROPPED** per the rule. Code is public but pretrained weights are **gated** (license-restricted, non-commercial). Without weights, "running" means pretraining on 2,500 h of EEG = days of compute. |
| EEGFormer / BrainBERT / EEG2Rep / AFTA | — | ❌ **DROPPED** — no public release found. |

## Why most of these are "partial"

Each of EEGPT / BENDR / FEMBA was pretrained on a *different* preprocessing
(channels, sampling rate, window length) from our TUAB_PREPROCESSED. The
pretrained weights therefore aren't directly transferable to our `[19, 2000]`
inputs without one of:

1. **Resample + zero-pad** our data to match their input shape — gives a number
   on a *modified* version of our pipeline (i.e. not directly comparable to our
   own JEPA numbers anymore).
2. **Re-train from scratch** with their architecture on our data — gives a "from
   scratch on our pipeline" number, like our existing supervised baselines, but
   loses the *pretraining* advantage that's the whole point of citing them.

BIOT escapes this because its 16-channel / 200 Hz / 10 s preprocessing matches
ours almost exactly — we only slice 19 → 18 channels, and the pretrained weights
apply directly.

For the others, both paths are valid research questions but require
significantly more adapter work than what fits in the hackathon. The cloned
repos are here so anyone can pick up the work post-hackathon.

## Setup commands (run on Dalia)

```bash
cd /lustre/work/vivatech-slightlyunawarefc/$USER/eb_jepa
git pull
bash baselines/external/setup_externals.sh
```

## Run commands

```bash
# BIOT vanilla (supervised, no pretraining) — matches LaBraM Tbl 1 "BIOT" line
# Expected: ~0.79 BAcc / ~0.87 AUROC at per-window
sbatch baselines/biot/run_biot.sbatch

# BIOT pretrained (6 EEG datasets) — fine-tune from PRETRAINED weights
PRETRAINED=$WORK/external/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt \
  sbatch baselines/biot/run_biot.sbatch
```

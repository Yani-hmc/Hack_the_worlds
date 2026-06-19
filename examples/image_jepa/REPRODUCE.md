# Reproducing Image-JEPA on CIFAR-10 (Phase 1)

This document is a ready-to-run reproduction of the **Image-JEPA on CIFAR-10** experiment
from the EB-JEPA paper. The model learns image representations by being invariant between two
augmented crops/views of the same image (a JEA — there is no predictor), regularized to avoid
collapse, and is evaluated with a **linear probe** on the CIFAR-10 test set.

> Paper-reported target: **~91% linear-probe top-1 accuracy on CIFAR-10**
> (best **91.02%** with SIGReg/BCS, **90.12%** with VICReg — see "Expected results" below).

All commands assume the repo root (`.../eb_jepa`) is the current directory. On DALIA, run
everything from `/lustre/work/vivatech-<team>/$USER/eb_jepa` after `source env.sh`.

---

## 0. What this experiment actually is

- **Encoder:** ResNet-18 (CIFAR-adapted: 3x3 stride-1 conv1, no maxpool, `fc=Identity`,
  feature dim **512**). ViT-S / ViT-B backbones are also available via `cfgs/transformers.yaml`.
- **Projector:** 3-layer MLP `Linear -> BN -> ReLU -> Linear -> BN -> ReLU -> Linear`
  (`features_dim -> proj_hidden_dim -> proj_hidden_dim -> proj_output_dim`). Loss is computed
  on the projected output. Disable with `model.use_projector=false`.
- **Regularizer (anti-collapse):** **VICReg** (`loss.type=vicreg`, variance + covariance, plus
  the MSE invariance term between the two views) OR **SIGReg** (`loss.type=bcs`, the BCS /
  Batched-Characteristic-Slicing Gaussianity loss with a single coefficient `loss.lmbd`).
- **Two views:** `ImageDataset(..., num_crops=2)` produces 2 augmented views per image
  (RandomResizedCrop scale (0.2,1.0), ColorJitter, Grayscale, Solarization, HFlip, Normalize).
- **Optimizer:** custom **LARS** (eta=0.02, momentum=0.9, clip_lr, exclude bias/norm),
  **WarmupCosine** LR schedule (`warmup_epochs=10`, `lr=0.3`, `min_lr=0`).
- **Linear probe:** a `nn.Linear(512, 10)` trained **online** on the **frozen** (detached)
  encoder features during the SSL run, and evaluated on the CIFAR-10 **test set every epoch**.
  Its test accuracy is logged as **`val_acc`** (and printed each `logging.log_every` epochs).
  **There is no separate eval script/command** — the linear-probe number is produced by the
  training run itself (see section 3).

Core code: `examples/image_jepa/main.py`, `dataset.py`, `eval.py`,
`cfgs/{default,sigreg,transformers}.yaml`; losses in `eb_jepa/losses.py`
(`VICRegLoss`, `BCS`).

---

## 1. TRAIN — local / interactive form

Default config = **ResNet-18 + VICReg, batch 256, 300 epochs** (`cfgs/default.yaml`):

```bash
python -m examples.image_jepa.main --fname examples/image_jepa/cfgs/default.yaml
```

To reproduce the paper's **best CIFAR-10 number (SIGReg/BCS, 91.02%)**, use the SIGReg config
(ResNet-18 + BCS, projector 2048x128, lmbd=10, 300 epochs):

```bash
python -m examples.image_jepa.main --fname examples/image_jepa/cfgs/sigreg.yaml
```

ViT backbone variant (optional):

```bash
python -m examples.image_jepa.main --fname examples/image_jepa/cfgs/transformers.yaml
# ViT-Base: append  model.type=vit_b
```

On DALIA, for a quick **interactive** debug run grab a GPU first (see DALIA_RUNBOOK §4A):

```bash
srun --partition=defq --reservation=Vivatech --gres=gpu:1 \
     --cpus-per-task=8 --time=00:30:00 --pty bash
# now ON a compute node:
uv run python -m examples.image_jepa.main \
     --fname examples/image_jepa/cfgs/default.yaml \
     optim.epochs=2 logging.log_wandb=false      # fast smoke test, see §6
```

---

## 2. TRAIN — DALIA / Slurm form (real run)

### A) Via the unified launcher (recommended; handles submitit, seeds, the aarch64 venv)

`image_jepa` is registered in `examples/launch_sbatch.py` with config
`examples/image_jepa/cfgs/default.yaml` and metric `val_acc`. The launcher applies the
cluster defaults from `env.sh` (`--partition=defq`, `--reservation=Vivatech`, 1 GPU, 8 CPUs).
By default it launches a **3-seed sweep** (seeds 1, 1000, 10000) of the default config:

```bash
python -m examples.launch_sbatch --example image_jepa
```

Single job / custom resources / config overrides:

```bash
# single job (no seed sweep):
python -m examples.launch_sbatch --example image_jepa --single

# reproduce the SIGReg best with a custom config + more CPUs + 8h:
python -m examples.launch_sbatch --example image_jepa --single \
     --fname examples/image_jepa/cfgs/sigreg.yaml --cpus-per-task 8 --time-min 480

# override any config key with dot notation (passed straight through):
python -m examples.launch_sbatch --example image_jepa --single --data.batch_size 256 --optim.epochs 300
```

> Do NOT pass `--mem`/`--mem-per-gpu` on DALIA — the scheduler rejects jobs that request
> memory explicitly (memory scales with CPU count). The launcher already omits it.

### B) Via the provided sbatch script (`run_cifar.sbatch`)

A self-contained batch script (mirrors `slurm_test.sh`) is provided at
`examples/image_jepa/run_cifar.sbatch`. It sources `env.sh`, `uv sync`s, and runs the
trainer. Submit it from the repo root:

```bash
# default VICReg config (300 epochs):
sbatch examples/image_jepa/run_cifar.sbatch

# reproduce the SIGReg best:
CFG=examples/image_jepa/cfgs/sigreg.yaml sbatch examples/image_jepa/run_cifar.sbatch

# fast smoke test (few epochs, no W&B):
EXTRA="optim.epochs=5 logging.log_wandb=false" sbatch examples/image_jepa/run_cifar.sbatch
```

It already hardcodes `--partition=defq --reservation=Vivatech --gres=gpu:1`
(plus `--cpus-per-task=8`, `--time=08:00:00`). Watch it with `sq` and tail with `log` (or
`log -f`). 300 epochs of ResNet-18 on CIFAR-10 on one GB200 is well within the 8h wall.

---

## 3. LINEAR-PROBE EVAL command

The linear probe is **trained and evaluated online inside the training run** — there is no
separate eval entry point. The number you care about is the CIFAR-10 **test-set** linear-probe
top-1 accuracy, logged each epoch as **`val_acc`** by `evaluate_linear_probe(...)` in
`eval.py` and pushed to W&B (and printed to stdout every `logging.log_every` epochs).

To "run the eval", you read `val_acc` from the same run:

- **W&B:** open the run for this experiment and read the `val_acc` curve (final / best epoch).
- **Stdout / Slurm log:** `log` (or `log <JOBID> -f`) and look at the `val_acc=...` lines
  printed by `log_epoch`. The **final-epoch / best `val_acc`** is the reproduced linear-probe
  accuracy.
- Checkpoints (`latest.pth.tar`, `epoch_*.pth.tar`) saved under
  `$EBJEPA_CKPTS/image_jepa/<sweep>/<exp>_seed<seed>/` store `linear_val_acc` and the
  `linear_probe_state_dict`, so the probe accuracy is also persisted with every checkpoint.

(If W&B is unavailable on the node, add `logging.log_wandb=false` — the probe still trains and
its `val_acc` is still printed/checkpointed.)

---

## 4. Expected results — what "reproduced / success" looks like

Source: the paper's CIFAR-10 results as reported in `examples/image_jepa/README.md`
(ResNet-18 backbone, 300 epochs, batch 256). **Do not invent other numbers** — these are the
ones to compare against.

| Setting | Config | Linear-probe top-1 (CIFAR-10) |
|---|---|---|
| **SIGReg (BCS), best** | `sigreg.yaml` (proj 2048x128, lmbd=10) | **91.02%** |
| SIGReg (BCS), avg non-collapsed | `sigreg.yaml` family | ~89.22% |
| **VICReg, best** | `default.yaml` family (proj 2048x1024, std=1 cov=100) | **90.12%** |
| VICReg, avg non-collapsed | `default.yaml` family | ~84.90% |
| No projector (either method) | `model.use_projector=false` | ~87.3–87.8% (≈ -2.5 to -3 pts) |

**Success criterion:** a single seed of `sigreg.yaml` (or `default.yaml`) trained for the full
300 epochs reaches a best/final `val_acc` of roughly **90–91%** on CIFAR-10. Landing in the
**~89–91%** band counts as a successful reproduction of the paper's ~91% headline number;
SIGReg should be the one that touches ~91%. A `val_acc` stuck near **10%** (chance) means the
representation **collapsed** — re-check the regularizer coefficients (see §5).

> Note: the shipped `cfgs/default.yaml` uses `std_coeff=1.0, cov_coeff=80.0`, which is a sane
> VICReg setting near the paper's best (`std=1, cov=100 -> 90.12%`); expect VICReg in the low
> 90s. For the documented **91.02%** peak, use **`sigreg.yaml`** as-is.

---

## 5. Key config knobs

From `cfgs/default.yaml` / `cfgs/sigreg.yaml`:

| Knob | Default (VICReg / SIGReg) | What it does |
|---|---|---|
| `model.type` | `resnet` (`vit_s`, `vit_b`) | backbone |
| `model.use_projector` | `true` | enable MLP projector (worth ~+2.5–3 pts) |
| `model.proj_hidden_dim` | `2048` | projector hidden width |
| `model.proj_output_dim` | `2048` (VICReg) / `128` (SIGReg) | projector output dim |
| `loss.type` | `vicreg` / `bcs` | regularizer: VICReg vs SIGReg |
| `loss.std_coeff` | `1.0` | VICReg variance weight (VICReg only) |
| `loss.cov_coeff` | `80.0` | VICReg covariance weight (VICReg only) |
| `loss.lmbd` | `10.0` | SIGReg/BCS single coefficient (BCS only) |
| `data.batch_size` | `256` | batch size (256 is the paper's setting) |
| `optim.epochs` | `300` | training length (300 = paper) |
| `optim.lr` | `0.3` | base LR (LARS + warmup-cosine) |
| `optim.warmup_epochs` | `10` | LR warmup |
| `training.dtype` | `bfloat16` (resnet) / `float16` (vit) | AMP precision |
| `meta.seed` | `42` | random seed |

**Failure mode to watch:** if the invariance/MSE term becomes negligible relative to the
regularizer (e.g. VICReg `std=100, cov=100`), training collapses to ~10%. Keep "logical"
coefficients (the defaults here are good).

---

## 6. Fast smoke tests (sanity-check the pipeline in minutes, NOT for accuracy)

These run a handful of epochs just to confirm the data loads, the model/loss/probe run, and
`val_acc` is produced. They will **not** reach 91% (too few epochs) — that's expected.

```bash
# Local / interactive — 2 epochs, no W&B:
python -m examples.image_jepa.main --fname examples/image_jepa/cfgs/default.yaml \
       optim.epochs=2 logging.log_wandb=false

# Even faster: tiny batch + 1 epoch (just exercises the loop):
python -m examples.image_jepa.main --fname examples/image_jepa/cfgs/sigreg.yaml \
       optim.epochs=1 data.batch_size=128 logging.log_wandb=false

# On DALIA via the sbatch script:
EXTRA="optim.epochs=5 logging.log_wandb=false" sbatch examples/image_jepa/run_cifar.sbatch
```

A successful smoke test prints rising `Acc` in the tqdm bar and a non-trivial `val_acc`
(well above 10% even after a couple of epochs).

---

## 7. Dataset: does CIFAR-10 auto-download? (internet on DALIA)

**Yes, it auto-downloads.** `main.py` calls:

```python
data_dir = os.environ.get("EBJEPA_DSETS")
CIFAR10(root=data_dir, train=True,  download=True, transform=None)
CIFAR10(root=data_dir, train=False, download=True, transform=get_val_transforms())
```

So on first run torchvision downloads `cifar-10-python.tar.gz` (~170 MB) into
**`$EBJEPA_DSETS`** (which `env.sh` defaults to `$WORK/datasets`, i.e.
`/lustre/work/vivatech-<team>/$USER/datasets`).

**DALIA compute nodes usually have NO internet**, so pre-stage the dataset once **on the login
node** (same pattern as the Moving-MNIST step in DALIA_RUNBOOK §3):

```bash
# On the LOGIN node, after `source env.sh` (so $EBJEPA_DSETS is set):
mkdir -p "$EBJEPA_DSETS"
python -c "import os; from torchvision.datasets import CIFAR10; CIFAR10(os.environ['EBJEPA_DSETS'], train=True, download=True); CIFAR10(os.environ['EBJEPA_DSETS'], train=False, download=True)"
```

Or, if the login node has `wget` but no torch handy, fetch the tarball directly into the same
folder (torchvision extracts it on first use):

```bash
mkdir -p "$EBJEPA_DSETS"
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P "$EBJEPA_DSETS/"
```

After pre-staging, `download=True` becomes a no-op on the compute node (torchvision sees the
extracted `cifar-10-batches-py/` and skips the download), so the offline job runs fine.

> If a shared/provided dataset folder already contains CIFAR-10, just point
> `EBJEPA_DSETS` at it (`export EBJEPA_DSETS=/path/to/shared/datasets`) before submitting and
> skip the download entirely.

---

## TL;DR

```bash
# 0. (login node, once) pre-stage CIFAR-10 offline:
python -c "import os;from torchvision.datasets import CIFAR10;CIFAR10(os.environ['EBJEPA_DSETS'],train=True,download=True);CIFAR10(os.environ['EBJEPA_DSETS'],train=False,download=True)"

# 1. TRAIN (best = SIGReg/BCS, 91.02%):
python -m examples.launch_sbatch --example image_jepa --single \
       --fname examples/image_jepa/cfgs/sigreg.yaml --time-min 480
#    (or: sbatch run_cifar.sbatch  /  CFG=.../sigreg.yaml sbatch examples/image_jepa/run_cifar.sbatch)

# 2. EVAL: read `val_acc` (CIFAR-10 test linear-probe) from W&B or the Slurm log — best ~91%.
```

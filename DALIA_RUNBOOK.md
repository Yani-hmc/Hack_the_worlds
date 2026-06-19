# Dalia + Slurm + Video-JEPA — Team Runbook

Quick, copy-pasteable path from a fresh SSH login to a running Moving-MNIST Video-JEPA
on the HackTheWorld(s) DALIA cluster. Commands are generic (`$USER`, `<team>`), so every
teammate can follow them.

## Cluster at a glance
- **Nodes:** 18 × `dalianvl`, each **4 × NVIDIA GB200** (185 GB VRAM each), 144 ARM Neoverse-V2 cores.
- **Login node is x86_64; compute nodes are aarch64** — the setup builds a separate venv per arch (handled for you).
- **Partition:** `defq` (2-day limit) · **Reservation:** `Vivatech` (pass `--reservation=Vivatech`).
- **Storage:** `/lustre/home` = 3 GB (code only!) · `/lustre/work/vivatech-<team>/$USER` = 10 TB (everything else).
- **Polite default per job:** ~36 CPUs + 1 GPU (VRAM is abundant; the bottleneck is data throughput, not the model).

---

## 0. Connect
```bash
ssh dalia        # alias preconfigured in ~/.ssh/config → tvasnier@dalia.idris.fr
```
First time only: accept the host fingerprint (`yes`). Drops you on the **login node** — never train here, always go through Slurm.

## 1. Clone the team repo + run setup
```bash
git clone <YOUR_GROUP_REPOSITORY_URL> eb_jepa
cd eb_jepa
bash setup.sh
```
`setup.sh` **relocates the repo to `/lustre/work/vivatech-<team>/$USER/eb_jepa`** (home quota is too small for git+venvs+caches), installs `uv`, syncs deps for both arches, and sets up Slurm account/QOS + (optional) W&B.

> ⚠️ Don't panic when the original clone "disappears" — it's intentional. A pointer `README.md` shows the new path.

## 2. Make env persistent + verify
```bash
cd /lustre/work/vivatech-<team>/$USER/eb_jepa
echo "source $(pwd)/env.sh" >> ~/.bashrc
source ~/.bashrc

sbatch slurm_test.sh        # runs pytest on a GPU node — confirms the env works
sq                          # watch it (PENDING → RUNNING → gone); then check the log
log                         # tail the most recent job's stdout
```

## 3. Pre-stage the Moving-MNIST dataset (do this on the LOGIN node)
Compute nodes usually have **no internet**, and the example auto-downloads ~800 MB on first run.
Fetch it once into your work datasets dir so jobs find it offline:
```bash
mkdir -p "$EBJEPA_DSETS"
wget https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy -P "$EBJEPA_DSETS/"
# ($EBJEPA_DSETS defaults to $WORK/datasets; set EBJEPA_DSETS to a shared folder if provided)
```

## 4. Run the Video-JEPA (Moving MNIST)
**A) Quick interactive debug** — grab a GPU and run a few steps by hand:
```bash
srun --partition=defq --reservation=Vivatech --gres=gpu:1 \
     --cpus-per-task=36 --time=00:30:00 --pty bash
# now ON a compute node:
uv run python -m examples.video_jepa.main \
     --fname examples/video_jepa/cfgs/default.yaml \
     model.steps=2 data.batch_size=128      # small override for a fast smoke test
```

**B) Real run via the launcher** (handles Slurm submission, seeds, sweeps):
```bash
python -m examples.launch_sbatch --example video_jepa
# override config or resources:
python -m examples.launch_sbatch --example video_jepa --cpus-per-task 36 --time-min 120
```
Default config is `cov_coeff=100, std_coeff=10, steps=4` (the paper's best VC setting → mAP ≈ 0.607).
Override any config key with dot notation, e.g. `model.steps=8`, `loss.cov_coeff=100`, `data.batch_size=128`.
Results/visualizations log to **W&B** (input frames, autoregressive rollout, digit-detection heatmaps).

**Key knobs:** `model.steps` (rollout horizon — higher predicts further but saturates ~8),
`loss.cov_coeff` / `loss.std_coeff` (anti-collapse VC loss), `model.henc`/`model.hpre`/`model.dstc` (encoder/predictor/repr dims).

## 5. Monitor (helper scripts, on PATH after `source env.sh`)
| Command | What it shows |
|---|---|
| `sq`        | **your** jobs, color-coded by state |
| `qall`      | all jobs on `defq` + per-user summary |
| `gpus`      | free/used GPUs and memory per node |
| `users`     | GPU/CPU usage per user |
| `log [JOBID] [-f]` | view/tail a job's stdout (no id = most recent; `-f` = follow) |

---

## Slurm cheat-sheet
| Command | Purpose |
|---|---|
| `srun … --pty bash` | interactive shell on a compute node (debug) |
| `sbatch script.sh` | submit a batch job (real runs) |
| `squeue --me` / `sq` | list your jobs (`R`=running, `PD`=pending) |
| `scancel <JOBID>` | kill a job (`scancel --me` = all yours) |
| `sinfo -p defq` | partition/node status |

Always pass on this cluster: `--partition=defq --reservation=Vivatech --gres=gpu:1`.
Account/QOS are auto-detected from your team allocation (override with `--account=…`).

## Gotchas (read before a long run)
- **Everything on `/lustre/work`, nothing in `$HOME`** — the 3 GB home quota blocks git/venvs/caches. `env.sh` points all caches to `$WORK/.cache`.
- **`--time` kills jobs at the limit** — a job vanishing at exactly its walltime is the limit, not a crash. **Checkpoint regularly** (ckpts go to `$WORK/checkpoints`).
- **Do NOT set `--mem`/`--mem-per-gpu`** — the DALIA scheduler *rejects* jobs that request memory explicitly; memory scales with CPU count instead.
- **`PD` (pending) is normal** when the cluster is busy — request only what you need (1 GPU, ~36 CPUs) to schedule sooner.
- **VRAM is not the bottleneck** (185 GB/GPU) — keep the GPU fed: enough dataloader CPUs, avoid logging overhead. Tune `data.batch_size` / `--cpus-per-task`.
- **Login node = x86_64, compute = aarch64** — run training through Slurm so it uses the aarch64 venv; the login node venv is CPU-only.

## Targets this weekend
- **`examples/video_jepa/`** — Moving-MNIST Video-JEPA (this runbook). Goal: train, get the autoregressive rollout + digit-detection mAP, then ablate `model.steps` / VC coeffs.
- **`examples/eeg/`** — the EEG example (`main.py` + `eval.py` + `cfgs/`); same launcher pattern: `python -m examples.launch_sbatch --example eeg`.
- Subject slides: `hackathon_guide/slides_hackathon_topic.pdf`, `slides-sujet-ESIEE.pdf`.

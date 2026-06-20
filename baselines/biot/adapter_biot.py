"""BIOT baseline adapter — plug OUR TUAB loader into the official BIOT model.

================================================================================
RUN ON DALIA (compute node, NOT login). Requires the official BIOT repo cloned
(see baselines/biot/README.md) on PYTHONPATH:

    git clone https://github.com/ycq091044/BIOT.git   # on the LOGIN node
    uv pip install linear-attention-transformer        # BIOT dep

    srun --partition=defq --reservation=Vivatech --gres=gpu:1 \
         --cpus-per-task=36 --time=02:00:00 --pty bash
    PYTHONPATH=/lustre/work/vivatech-<team>/$USER/BIOT:$PYTHONPATH \
      uv run python baselines/biot/adapter_biot.py \
        --epochs 20 --batch-size 128 --lr 1e-3 --n-channels 19
================================================================================

Paper-reported TUAB (BIOT Table 3, arXiv:2305.10351, ⟨verify⟩):
    Balanced-Acc 0.7959 | AUROC 0.8815 | AUC-PR 0.8792

This adapter imports the BIOT model from the CLONED repo (not vendored here),
feeds it OUR dataset windows (19ch, 200Hz, 10s = [B,19,2000], z-scored,
patient-disjoint), trains supervised on TUAB train patients, evaluates
patient-disjoint on eval, and prints accuracy / balanced-acc / precision /
recall / F1 / AUROC per-window AND aggregated to recording level (to match our
JEPA frozen probe).

The BIOT import is GUARDED so this file `py_compile`s without the repo present;
it only needs BIOT at RUN time on Dalia.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset


def build_biot(n_channels, n_classes=2, sampling_rate=200):
    """Import BIOTClassifier from the cloned official repo (run-time only)."""
    try:
        from model import BIOTClassifier  # provided by github.com/ycq091044/BIOT
    except ImportError as e:  # pragma: no cover - only triggers off-cluster
        raise ImportError(
            "BIOTClassifier not found. Clone https://github.com/ycq091044/BIOT "
            "and add it to PYTHONPATH (see baselines/biot/README.md)."
        ) from e
    # BIOT default hidden/heads/layers per the repo's TUAB config.
    return BIOTClassifier(
        n_classes=n_classes,
        n_channels=n_channels,
        n_fft=200,            # 1 s STFT window at 200 Hz
        hop_length=100,
    )


# --------------------------------------------------------------------------- #
# Window-level view — shared helper supporting n_windows=-1 (ALL non-overlapping
# windows per recording, literature protocol) and n_windows=16 (legacy).
# --------------------------------------------------------------------------- #
from eb_jepa.datasets.eeg.window_dataset import WindowDataset  # noqa: E402


def _metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auroc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "n": int(len(y_true)),
    }


@torch.no_grad()
def evaluate(model, loader, device, n_channels):
    model.eval()
    probs, labels, recs, oks = [], [], [], []
    for x, y, rec, ok in loader:
        x = x[:, :n_channels].to(device)          # [B, C, T]
        p = F.softmax(model(x), dim=1)[:, 1].cpu().numpy()
        probs.append(p); labels.append(y.numpy())
        recs.append(rec.numpy()); oks.append(ok.numpy())
    probs = np.concatenate(probs); labels = np.concatenate(labels)
    recs = np.concatenate(recs); oks = np.concatenate(oks).astype(bool)
    probs, labels, recs = probs[oks], labels[oks], recs[oks]

    win = _metrics(labels, (probs >= 0.5).astype(int), probs)

    rec_prob, rec_lab = {}, {}
    for p, l, r in zip(probs, labels, recs):
        rec_prob.setdefault(r, []).append(p)
        rec_lab[r] = l
    r_ids = sorted(rec_prob)
    r_prob = np.array([np.mean(rec_prob[r]) for r in r_ids])
    r_lab = np.array([rec_lab[r] for r in r_ids])
    rec = _metrics(r_lab, (r_prob >= 0.5).astype(int), r_prob)
    return win, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--n-channels", type=int, default=19)
    ap.add_argument("--n-windows", type=int, default=-1,
                    help="-1 = ALL non-overlapping (literature protocol); "
                         "16 = legacy evenly-spaced subsample")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrained-ckpt", default=None,
                    help="Optional path to a BIOT pretrained checkpoint "
                    "(e.g. $WORK/external/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt). "
                    "When set, loads the encoder weights then fine-tunes end-to-end.")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[biot] device={device} n_channels={args.n_channels}", flush=True)

    train_ds = WindowDataset("train", args.n_windows)
    eval_ds = WindowDataset("eval", args.n_windows)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0)
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0)

    model = build_biot(args.n_channels, n_classes=2, sampling_rate=200).to(device)
    if args.pretrained_ckpt:
        # Pretrained checkpoints in BIOT/pretrained-models/ are raw BIOTEncoder
        # state dicts (no "biot." prefix). Load into model.biot directly.
        print(f"[biot] loading pretrained encoder from {args.pretrained_ckpt}", flush=True)
        state = torch.load(args.pretrained_ckpt, map_location=device, weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.biot.load_state_dict(state, strict=False)
        print(f"[biot]   loaded; missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y, _, ok in train_loader:
            ok = ok.bool()
            if ok.sum() == 0:
                continue
            x = x[ok][:, :args.n_channels].to(device)
            y = y[ok].to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            running = loss.item()
        print(f"[biot] epoch {epoch} loss={running:.4f}", flush=True)

    win, rec = evaluate(model, eval_loader, device, args.n_channels)
    print("[biot] EVAL (patient-disjoint)")
    print("  per-window     :", win)
    print("  per-recording  :", rec, "  <-- compare to JEPA frozen probe")


if __name__ == "__main__":
    main()

"""ShallowConvNet (Schirrmeister 2017) baseline on TUAB — supervised, end-to-end.

================================================================================
RUN ON DALIA (compute node, NOT login):
    srun --partition=defq --reservation=Vivatech --gres=gpu:1 \
         --cpus-per-task=36 --time=00:30:00 --pty bash
    uv run python -m baselines.shallowconvnet.train_shallowconvnet_tuab --epochs 1

    # full run:
    uv run python -m baselines.shallowconvnet.train_shallowconvnet_tuab \
         --epochs 30 --batch-size 256 --lr 6.25e-4
================================================================================

ShallowFBCSPNet (= the "ShallowConvNet" of Schirrmeister 2017, arXiv:1703.05051 /
arXiv:1708.08012). We use the CANONICAL implementation `braindecode.models.ShallowFBCSPNet`
(BSD-3-Clause) -- maintained by Robin Schirrmeister himself, the paper's first author.
Architecture: temporal Conv2d (1,25) -> spatial Conv2d (C,1) -> BN -> square ->
AvgPool(1,75) stride 15 -> log -> Dropout -> Conv2d classifier -> logits. The square/log
"pool" emulates a band-power feature; designed for ~250 Hz, works on our 200 Hz
[19, 2000] windows. No in-file reimplementation -- using upstream removes reimpl drift.

Reuses OUR dataset (`eb_jepa.datasets.eeg.dataset`) — exact spec:
    19 channels, 200 Hz, 10 s -> [B, 19, 2000], per-channel z-scored,
    patient-disjoint train/eval. Trains per-window; reports metrics per-window AND
    aggregated to recording level (mean window-prob) to match our JEPA probe.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from braindecode.models import ShallowFBCSPNet

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset


# --------------------------------------------------------------------------- #
# Window-level view — shared helper supports n_windows=-1 (ALL non-overlapping
# windows per recording, literature protocol).
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
def evaluate(model, loader, device):
    model.eval()
    probs, labels, recs, oks = [], [], [], []
    for x, y, rec, ok in loader:
        p = F.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
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
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=6.25e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--n-windows-train", type=int, default=16,
                    help="windows/rec at TRAIN (16 = fast; -1 = all non-overlapping)")
    ap.add_argument("--n-windows-eval", type=int, default=-1,
                    help="windows/rec at EVAL (-1 = all non-overlapping, literature protocol)")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[shallowconvnet] device={device}", flush=True)

    train_ds = WindowDataset("train", args.n_windows_train)
    eval_ds = WindowDataset("eval", args.n_windows_eval)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0)
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0)

    # Canonical implementation by Robin Schirrmeister himself (BSD-3-Clause).
    # ShallowFBCSPNet expects [B, n_chans, n_times] -- our WindowDataset yields that.
    model = ShallowFBCSPNet(n_chans=19, n_outputs=2, n_times=2000).to(device)
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
            x, y = x[ok].to(device), y[ok].to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            running = loss.item()
        print(f"[shallowconvnet] epoch {epoch} loss={running:.4f}", flush=True)

    win, rec = evaluate(model, eval_loader, device)
    print("[shallowconvnet] EVAL (patient-disjoint)")
    print("  per-window     :", win)
    print("  per-recording  :", rec, "  <-- compare to JEPA frozen probe")


if __name__ == "__main__":
    main()

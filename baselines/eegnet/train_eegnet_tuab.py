"""EEGNet baseline on TUAB (normal vs abnormal) — supervised, end-to-end.

================================================================================
RUN ON DALIA (compute node, NOT login):
    # interactive smoke test:
    srun --partition=defq --reservation=Vivatech --gres=gpu:1 \
         --cpus-per-task=36 --time=00:30:00 --pty bash
    uv run python -m baselines.eegnet.train_eegnet_tuab --epochs 1

    # full run:
    uv run python -m baselines.eegnet.train_eegnet_tuab \
         --epochs 30 --batch-size 256 --lr 1e-3
================================================================================

This is a literature baseline for Phase 0. It reuses OUR dataset
(`eb_jepa.datasets.eeg.dataset`) so the input is exactly our spec:
    19 channels, 200 Hz, 10 s windows -> [B, 19, 2000], per-channel z-scored,
    patient-disjoint train/eval.

EEGNet (Lawhern et al., 2018, arXiv:1611.08024). We use the CANONICAL PyTorch port
`braindecode.models.EEGNetv4` (BSD-3-Clause), maintained by the braindecode team and
used in every PyTorch EEG paper. Lawhern's official repo (vlawhern/arl-eegmodels) is
Keras/TF; EEGNetv4 mirrors its architecture 1:1 in PyTorch. No in-file reimplementation
here -- using the upstream implementation removes "reimpl drift" as a confound.

We train PER-WINDOW (one [19,2000] window = one labelled example, label inherited
from its recording), then report metrics BOTH per-window AND aggregated to
recording level (mean window-probability per recording) to match our JEPA's
recording-level frozen probe in `examples/eeg/eval.py`. The recording-level row is
the apples-to-apples comparison.

NOTE: trains on labelled windows. We build a window-level view on top of the
probe-mode dataset (which yields [N,C,T] per recording) so no new EDF I/O code is
needed and the patient-disjoint split is preserved.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from braindecode.models import EEGNetv4

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset


# --------------------------------------------------------------------------- #
# Window-level dataset view on top of the probe-mode recordings
# --------------------------------------------------------------------------- #
class WindowDataset(torch.utils.data.Dataset):
    """Flatten probe-mode recordings into labelled [C, T] windows.

    Each probe item is [N, C, T] windows + label; we expand to N windows that all
    inherit the recording label. Recording index is kept so we can aggregate
    predictions back to recording level for the apples-to-apples metric.
    """

    def __init__(self, split, n_windows=16):
        cfg = EEGConfig(split=split, mode="probe", n_windows=n_windows)
        self.base = EEGDataset(cfg)
        self.n_windows = n_windows

    def __len__(self):
        return len(self.base) * self.n_windows

    def __getitem__(self, idx):
        rec_i, win_i = divmod(idx, self.n_windows)
        wins, label, ok = self.base[rec_i]      # wins: [N, C, T] tensor
        x = wins[win_i]                          # [C, T]
        return x, int(label), int(rec_i), int(bool(ok))


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
    probs, labels, recs = probs[oks], labels[oks], recs[oks]   # drop unreadable

    win = _metrics(labels, (probs >= 0.5).astype(int), probs)

    # aggregate to recording level: mean window-prob per recording
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
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5,
                    help="0.5 is the canonical EEGNet value for cross-subject tasks")
    ap.add_argument("--n-windows", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eegnet] device={device}", flush=True)

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

    # Canonical PyTorch port from braindecode (BSD-3-Clause). EEGNetv4 expects
    # [B, n_chans, n_times] -- our WindowDataset yields exactly that shape.
    # drop_prob is braindecode's name for the dropout parameter; honors the
    # --dropout CLI flag (0.5 = canonical cross-subject default per Yani 2cf7a0e).
    model = EEGNetv4(n_chans=19, n_outputs=2, n_times=2000,
                     drop_prob=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # class weights handle TUAB's mild normal/abnormal imbalance
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
        print(f"[eegnet] epoch {epoch} loss={running:.4f}", flush=True)

    win, rec = evaluate(model, eval_loader, device)
    print("[eegnet] EVAL (patient-disjoint)")
    print("  per-window     :", win)
    print("  per-recording  :", rec, "  <-- compare to JEPA frozen probe")


if __name__ == "__main__":
    main()

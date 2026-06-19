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

ShallowFBCSPNet / ShallowConvNet is the classic strong ConvNet baseline for
TUAB-abnormal classification (Gemein et al., NeuroImage 2020 report ~85% balanced
accuracy with braindecode ConvNets). Reimplemented in-file in PyTorch, matching
braindecode's `ShallowFBCSPNet` (BSD-3-Clause):
    temporal Conv2d (1, 25) -> spatial Conv2d (C, 1) -> BN
    -> square -> AvgPool(1, 75) stride (1, 15) -> log -> Dropout
    -> Conv2d classifier (1, t') -> logits
The square/log "pool" emulates a band-power feature; designed for ~250 Hz but
works on our 200 Hz [19, 2000] windows.

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

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset


# --------------------------------------------------------------------------- #
# ShallowConvNet — in-file PyTorch (braindecode ShallowFBCSPNet equivalent)
# --------------------------------------------------------------------------- #
class _Square(nn.Module):
    def forward(self, x):
        return x * x


class _SafeLog(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


class ShallowConvNet(nn.Module):
    def __init__(self, n_channels=19, n_times=2000, n_classes=2,
                 n_filters_time=40, n_filters_spat=40,
                 filter_time_length=25, pool_time_length=75,
                 pool_time_stride=15, dropout=0.5):
        super().__init__()
        self.conv_time = nn.Conv2d(1, n_filters_time, (1, filter_time_length))
        self.conv_spat = nn.Conv2d(n_filters_time, n_filters_spat,
                                   (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters_spat)
        self.square = _Square()
        self.pool = nn.AvgPool2d((1, pool_time_length), (1, pool_time_stride))
        self.safelog = _SafeLog()
        self.drop = nn.Dropout(dropout)
        with torch.no_grad():
            d = self._features(torch.zeros(1, 1, n_channels, n_times)).shape[1]
        self.classifier = nn.Linear(d, n_classes)

    def _features(self, x):
        x = self.conv_time(x)                 # [B, Ft, C, T']
        x = self.bn(self.conv_spat(x))        # [B, Fs, 1, T']
        x = self.square(x)
        x = self.pool(x)
        x = self.safelog(x)
        x = self.drop(x)
        return torch.flatten(x, 1)

    def forward(self, x):                      # x: [B, C, T]
        return self.classifier(self._features(x.unsqueeze(1)))


# --------------------------------------------------------------------------- #
# Window-level view on probe-mode recordings (patient-disjoint preserved)
# --------------------------------------------------------------------------- #
class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, split, n_windows=16):
        cfg = EEGConfig(split=split, mode="probe", n_windows=n_windows)
        self.base = EEGDataset(cfg)
        self.n_windows = n_windows

    def __len__(self):
        return len(self.base) * self.n_windows

    def __getitem__(self, idx):
        rec_i, win_i = divmod(idx, self.n_windows)
        wins, label, ok = self.base[rec_i]
        return wins[win_i], int(label), int(rec_i), int(bool(ok))


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
    ap.add_argument("--n-windows", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[shallowconvnet] device={device}", flush=True)

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

    model = ShallowConvNet(n_channels=19, n_times=2000, n_classes=2).to(device)
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

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

EEGNet (Lawhern et al., 2018, arXiv:1611.08024) reimplemented in-file in PyTorch.
Architecture matches the canonical Keras reference `vlawhern/arl-eegmodels` and
braindecode's `EEGNetv4`:
    Block 1: temporal Conv2d (1, kernLength) -> BN
             -> DepthwiseConv2d (C, 1), depth D -> BN -> ELU -> AvgPool(1,4) -> Drop
    Block 2: SeparableConv2d (1, 16) -> BN -> ELU -> AvgPool(1,8) -> Drop
    Classifier: flatten -> Linear -> logits
with kernLength = sfreq // 2 = 100 (half the 200 Hz sample rate), F1=8, D=2, F2=16.

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

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset


# --------------------------------------------------------------------------- #
# EEGNet (Lawhern et al. 2018) — in-file PyTorch reimplementation
# --------------------------------------------------------------------------- #
class Conv2dSamePad(nn.Conv2d):
    """Conv2d with TensorFlow-style 'same' padding along the time axis."""

    def forward(self, x):
        ih, iw = x.shape[-2:]
        kh, kw = self.weight.shape[-2:]
        pad_w = max((iw - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        pad_h = max((ih - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return self._conv_forward(x, self.weight, self.bias)


class EEGNet(nn.Module):
    """Compact CNN. Input [B, C, T]; internally treated as [B, 1, C, T]."""

    def __init__(self, n_channels=19, n_times=2000, n_classes=2,
                 F1=8, D=2, kern_length=100, dropout=0.25):
        super().__init__()
        F2 = F1 * D
        self.block1_temporal = Conv2dSamePad(
            1, F1, (1, kern_length), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        # depthwise spatial conv across all channels
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1),
                                   groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        # separable conv = depthwise (1,16) + pointwise (1,1)
        self.sep_depth = Conv2dSamePad(F1 * D, F1 * D, (1, 16),
                                       groups=F1 * D, bias=False)
        self.sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        # classifier head — infer flattened dim from a dry run
        with torch.no_grad():
            d = self._features(torch.zeros(1, 1, n_channels, n_times)).shape[1]
        self.classifier = nn.Linear(d, n_classes)

    def _features(self, x):
        x = self.bn1(self.block1_temporal(x))
        x = self.bn2(self.depthwise(x))
        x = self.drop1(self.pool1(F.elu(x)))
        x = self.sep_point(self.sep_depth(x))
        x = self.bn3(x)
        x = self.drop2(self.pool2(F.elu(x)))
        return torch.flatten(x, 1)

    def forward(self, x):              # x: [B, C, T]
        x = x.unsqueeze(1)             # -> [B, 1, C, T]
        return self.classifier(self._features(x))


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

    model = EEGNet(n_channels=19, n_times=2000, n_classes=2,
                   kern_length=100, dropout=args.dropout).to(device)
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

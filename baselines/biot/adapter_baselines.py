"""Official BIOT-repo baseline ZOO adapter — run ALL 6 architectures from the
exact upstream code (github.com/ycq091044/BIOT) on OUR TUAB_PREPROCESSED pipeline.

This generalises `adapter_biot.py` (which only ran BIOTClassifier) to the full set
of models that the official `run_binary_supervised.py` instantiates:

    SPaRCNet · ContraWR · CNNTransformer · FFCL · STTransformer · BIOT

Every constructor below is COPIED VERBATIM from the official
`run_binary_supervised.py::supervised()` (TUAB branch), so the models are byte-for-
byte the upstream ones — only the *data loader* (our patient-disjoint TUAB windows)
and the *eval protocol* (per-window + per-recording, to match our JEPA probe) are ours.

Why this matters: it turns the LaBraM-Table-1 literature rows
    SPaRCNet 0.7896 · ContraWR 0.7746 · CNN-Transformer 0.7777 · FFCL 0.7848 ·
    ST-Transformer 0.7966 · BIOT 0.7959
from *cited numbers* into *numbers we reproduced ourselves from the authors' code*
on our exact preprocessing — an honest apples-to-apples baseline panel.

RUN ON DALIA (compute node). The BIOT repo must be on PYTHONPATH:

    PYTHONPATH=$WORK/external/BIOT:$PYTHONPATH \
      uv run python -m baselines.biot.adapter_baselines \
        --model STTransformer --epochs 20 --batch-size 128 --n-channels 19

The `from model import ...` is GUARDED so this file py_compiles without the repo.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset

MODELS = ["SPaRCNet", "ContraWR", "CNNTransformer", "FFCL", "STTransformer", "BIOT"]


def build_model(name, n_channels, sample_length, n_classes=2):
    """Construct one of the 6 official BIOT-repo models.

    `sample_length` is the number of timesteps (T = sampling_rate * seconds = 2000).
    The constructor args mirror run_binary_supervised.py::supervised() exactly, with
    the official TUAB defaults: token_size/n_fft=200, hop_length=100, steps=20.
    """
    try:
        from model import (SPaRCNet, ContraWR, CNNTransformer, FFCL,
                           STTransformer, BIOTClassifier)
    except ImportError as e:  # pragma: no cover - only triggers off-cluster
        raise ImportError(
            "BIOT model.py not found. Clone https://github.com/ycq091044/BIOT and "
            "add it to PYTHONPATH (see baselines/biot/README.md)."
        ) from e

    if name == "SPaRCNet":
        return SPaRCNet(in_channels=n_channels, sample_length=sample_length,
                        n_classes=n_classes, block_layers=4, growth_rate=16,
                        bn_size=16, drop_rate=0.5, conv_bias=True, batch_norm=True)
    if name == "ContraWR":
        return ContraWR(in_channels=n_channels, n_classes=n_classes,
                        fft=200, steps=20)
    if name == "CNNTransformer":
        return CNNTransformer(in_channels=n_channels, n_classes=n_classes,
                              fft=200, steps=20, dropout=0.2, nhead=4, emb_size=256)
    if name == "FFCL":
        return FFCL(in_channels=n_channels, n_classes=n_classes, fft=200, steps=20,
                    sample_length=sample_length, shrink_steps=20)
    if name == "STTransformer":
        return STTransformer(emb_size=256, depth=4, n_classes=n_classes,
                             channel_legnth=sample_length, n_channels=n_channels)
    if name == "BIOT":
        return BIOTClassifier(n_classes=n_classes, n_channels=n_channels,
                              n_fft=200, hop_length=100)
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


# --------------------------------------------------------------------------- #
# Window-level view: import the shared WindowDataset that supports both
#   n_windows=16  (legacy evenly-spaced) AND
#   n_windows=-1 (ALL non-overlapping windows, literature-canonical eval)
# --------------------------------------------------------------------------- #
from eb_jepa.datasets.eeg.window_dataset import WindowDataset  # noqa: E402


def _metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 average_precision_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auroc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "auc_pr": round(float(average_precision_score(y_true, y_prob)), 4),
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
    ap.add_argument("--model", choices=MODELS, default="STTransformer")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--n-channels", type=int, default=19)
    ap.add_argument("--sample-length", type=int, default=2000,
                    help="timesteps per window (sampling_rate*seconds = 200*10)")
    ap.add_argument("--n-windows-train", type=int, default=16,
                    help="windows/rec at TRAIN (16 = fast, organizers' default; "
                         "-1 = ALL non-overlapping, much slower)")
    ap.add_argument("--n-windows-eval", type=int, default=-1,
                    help="windows/rec at EVAL (-1 = ALL non-overlapping, "
                         "literature protocol — the fair per-recording vote; 16 = legacy)")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.model
    print(f"[{tag}] device={device} n_channels={args.n_channels} "
          f"sample_length={args.sample_length}", flush=True)

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

    model = build_model(args.model, args.n_channels, args.sample_length).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] params={n_par/1e6:.2f}M", flush=True)
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
        print(f"[{tag}] epoch {epoch} loss={running:.4f}", flush=True)

    win, rec = evaluate(model, eval_loader, device, args.n_channels)
    print(f"[{tag}] EVAL (patient-disjoint)")
    print("  per-window     :", win)
    print("  per-recording  :", rec, "  <-- compare to JEPA frozen probe")
    print(f"=== DONE {tag} ===")


if __name__ == "__main__":
    main()

# TUEV 6-class event probe on a FROZEN TUAB-pretrained encoder (transfer test).
# Run (Dalia):
#   python -m examples.eeg.tuev_probe --ckpt <.../latest.pth.tar>
#   python -m examples.eeg.tuev_probe --ckpt <...> --random   # untrained-encoder floor
"""Does a frozen self-supervised encoder (pretrained for TUAB *binary abnormality*)
also linearly separate the 6 TUEV *event* classes? — a representation transfer test.

TUEV samples are .npz with:
  sig     [19, T]      float32   19 ch @ 200 Hz
  lab_sec [T // 200]   int64     one event label per second (6 classes)
  subject  str
We cut each recording into non-overlapping W-second windows, label each by the
MAJORITY of its per-second labels, z-score per channel (matching the TUAB pipeline),
encode with the FROZEN encoder, and fit a class-balanced multinomial logistic probe.
train/ vs eval/ are subject-disjoint (provided by the dataset).

Metrics: accuracy, balanced-accuracy, macro-F1, weighted-F1, Cohen's kappa — the
imbalance-aware metrics the TUEV literature (BIOT, LaBraM) reports.
CAVEAT: our fixed-window + majority-label framing differs from the BIOT event-centered
5 s protocol, so numbers are indicative of representation quality, not a 1:1 SOTA match.
"""
import argparse
import glob
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from examples.eeg.main import build_encoder

SFREQ = 200


def _zscore(x):  # [C, T] per-channel (same as the TUAB loader)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    return (x - mu) / sd


def _label_hist(files):
    """Per-second label histogram (labels only, cheap) -> dict + background class."""
    hist = {}
    for f in files:
        lab = np.asarray(np.load(f, allow_pickle=True)["lab_sec"]).astype(np.int64)
        v, c = np.unique(lab[lab >= 0], return_counts=True)
        for vi, ci in zip(v.tolist(), c.tolist()):
            hist[vi] = hist.get(vi, 0) + ci
    bg = max(hist, key=hist.get)                 # background = most frequent class
    return hist, bg


@torch.no_grad()
def extract(encoder, files, device, window_sec, bg_label, p_bg, rng):
    """EVENT-CENTERED windows: one window centered on each labeled second, with the
    background class subsampled (keep prob p_bg) so the 6 classes are represented.
    This mirrors the TUEV literature's event-segment protocol (vs. drowning in BCKG)."""
    half = (window_sec * SFREQ) // 2
    X, y, buf = [], [], []

    def flush():
        ws = torch.from_numpy(np.stack([b[0] for b in buf])).float().to(device)
        z = encoder.represent(ws).cpu().numpy()
        for k in range(len(buf)):
            X.append(z[k]); y.append(buf[k][1])
        buf.clear()

    for f in files:
        d = np.load(f, allow_pickle=True)
        sig = d["sig"].astype(np.float32)                  # [19, T]
        lab = np.asarray(d["lab_sec"]).astype(np.int64)    # [S]
        T = sig.shape[1]
        for s in range(len(lab)):
            label = int(lab[s])
            if label < 0:
                continue
            if label == bg_label and rng.random() > p_bg:   # subsample background
                continue
            c = s * SFREQ + SFREQ // 2                       # center sample of second s
            if c - half < 0 or c + half > T:
                continue
            w = _zscore(sig[:, c - half:c + half])
            buf.append((w, label))
            if len(buf) >= 256:
                flush()
    if buf:
        flush()
    return np.stack(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="TUAB-pretrained SSL checkpoint")
    ap.add_argument("--data-root",
                    default="/lustre/work/pdl17890/udl806719/datasets/Neuro/TUEV_PREP200")
    ap.add_argument("--window-sec", type=int, default=5)
    ap.add_argument("--max-files", type=int, default=None, help="debug: limit files")
    ap.add_argument("--random", action="store_true",
                    help="use an UNTRAINED encoder (random-feature floor)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    encoder = build_encoder(cfg.model).to(device)
    if not args.random:
        encoder.load_state_dict(state["encoder"])    # else keep random init = floor
    encoder.eval()
    tag = "RANDOM-floor" if args.random else "trained"

    import random
    tr_files = sorted(glob.glob(os.path.join(args.data_root, "train", "*.npz")))
    ev_files = sorted(glob.glob(os.path.join(args.data_root, "eval", "*.npz")))
    if args.max_files:
        tr_files, ev_files = tr_files[:args.max_files], ev_files[:args.max_files]
    hist, bg = _label_hist(tr_files)
    nonbg = sum(c for k, c in hist.items() if k != bg)
    p_bg = min(1.0, nonbg / max(1, hist[bg]))      # keep background count ~= total non-bg
    print(f"[tuev] train label hist={hist} | background={bg} | keep-bg p={p_bg:.4f}", flush=True)

    print(f"[tuev] ({tag}) extracting TRAIN windows (event-centered, bg subsampled)...", flush=True)
    Xtr, ytr = extract(encoder, tr_files, device, args.window_sec, bg, p_bg, random.Random(0))
    print(f"[tuev] ({tag}) extracting EVAL windows...", flush=True)
    Xev, yev = extract(encoder, ev_files, device, args.window_sec, bg, p_bg, random.Random(1))

    # remap labels to 0..K-1 (TUEV may be 1..6)
    classes = sorted(set(ytr.tolist()) | set(yev.tolist()))
    remap = {c: i for i, c in enumerate(classes)}
    ytr = np.array([remap[v] for v in ytr])
    yev = np.array([remap[v] for v in yev])
    print(f"[tuev] classes={classes} | train n={len(ytr)} dist={np.bincount(ytr).tolist()}"
          f" | eval n={len(yev)} dist={np.bincount(yev).tolist()}", flush=True)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 cohen_kappa_score, f1_score)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xev))
    print(f"[tuev][{tag}]", {
        "accuracy": round(float(accuracy_score(yev, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(yev, pred)), 4),
        "macro_f1": round(float(f1_score(yev, pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(yev, pred, average="weighted")), 4),
        "cohen_kappa": round(float(cohen_kappa_score(yev, pred)), 4),
        "n_train": int(len(ytr)), "n_eval": int(len(yev)), "n_classes": len(classes),
    })


if __name__ == "__main__":
    main()

"""EEG — downstream evaluation (the patient-disjoint abnormality probe).

The feature-extraction harness is provided: per recording, encode N evenly-spaced
10 s windows with the FROZEN encoder and mean-pool them into ONE embedding. What
you implement (`# TODO`) is the probe + metric.

GOLDEN RULE — patient-disjoint split: fit the probe on `train` patients, score on
`eval` patients (no subject overlap). A probe that scores well *within* a subject
but collapses across subjects is measuring identity, not pathology — so the held-
out-patient number is the only one that answers the transferability question.

Run:  python -m examples.eeg.eval --ckpt <.../latest.pth.tar>
"""
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.main import build_encoder


@torch.no_grad()
def extract_features(encoder, split, device):
    """Provided: frozen encoder -> [N_rec, D] recording-level features + labels.

    One embedding per recording: encode its N windows and mean-pool them.
    """
    ds = EEGDataset(EEGConfig(split=split, mode="probe"))
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=16,
                                         pin_memory=True)
    X, y = [], []
    for wins, labels, ok in loader:          # wins: [B, N, C, T]
        B, N = wins.shape[0], wins.shape[1]
        flat = wins.reshape(B * N, *wins.shape[2:]).to(device, non_blocking=True)
        z = encoder.represent(flat).reshape(B, N, -1).mean(dim=1)  # [B, D]
        z = z.cpu().numpy()
        for k in range(B):
            if bool(ok[k]):                  # drop unreadable recordings
                X.append(z[k]); y.append(int(labels[k]))
    return np.stack(X), np.array(y)


# --------------------------------------------------------------------------- #
# PROBE + METRIC  — # TODO
# --------------------------------------------------------------------------- #
def probe(Xtr, ytr, Xev, yev):
    """Patient-disjoint frozen-feature probe.

    Standardize on TRAIN stats only (no leakage), fit a class-balanced
    LogisticRegression on the train-patient embeddings, and score on the
    held-out-patient eval embeddings. normal=0 vs abnormal=1.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)

    scaler = StandardScaler().fit(Xtr)             # TRAIN stats only
    Xtr_s, Xev_s = scaler.transform(Xtr), scaler.transform(Xev)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr_s, ytr)

    pred = clf.predict(Xev_s)
    proba = clf.predict_proba(Xev_s)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(yev, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(yev, pred)), 4),
        "precision": round(float(precision_score(yev, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(yev, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(yev, pred, zero_division=0)), 4),
        "auroc": round(float(roc_auc_score(yev, proba)), 4),
        "n_train": int(len(ytr)), "n_eval": int(len(yev)),
        "frac_abnormal_eval": round(float(np.mean(yev)), 4),
    }


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    encoder = build_encoder(cfg.model).to(device)
    encoder.load_state_dict(state["encoder"]); encoder.eval()

    print("[eeg-eval] extracting TRAIN embeddings (fit set)...", flush=True)
    Xtr, ytr = extract_features(encoder, "train", device)
    print("[eeg-eval] extracting EVAL embeddings (held-out patients)...", flush=True)
    Xev, yev = extract_features(encoder, "eval", device)
    print("[eeg-eval]", probe(Xtr, ytr, Xev, yev))


if __name__ == "__main__":
    main()

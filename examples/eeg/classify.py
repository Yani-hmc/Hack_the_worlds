# Phase 3b — MLP-head classifier on frozen SSL features (patient-disjoint).
# Run on Dalia (GPU + dataset required):
#   python -m examples.eeg.classify --ckpt <.../latest.pth.tar>
#   python -m examples.eeg.classify --ckpt <.../latest.pth.tar> --head both
# Trains a small MLP (and/or LogisticRegression) on the FROZEN recording-level
# embeddings and reports accuracy/balanced-acc/precision/recall/F1/AUROC.
# This is a thin CLI around eval.mlp_probe / eval.probe; the same logic is also
# reachable via `python -m examples.eeg.eval --probe mlp`.
"""Standalone MLP-head classifier over frozen EEG SSL features."""
import argparse

import torch
from omegaconf import OmegaConf

from examples.eeg.main import build_encoder
from examples.eeg.eval import extract_features, probe, mlp_probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--head", default="mlp", choices=["mlp", "logreg", "both"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    encoder = build_encoder(cfg.model).to(device)
    encoder.load_state_dict(state["encoder"]); encoder.eval()

    print("[classify] extracting TRAIN embeddings...", flush=True)
    Xtr, ytr = extract_features(encoder, "train", device)
    print("[classify] extracting EVAL embeddings (held-out patients)...", flush=True)
    Xev, yev = extract_features(encoder, "eval", device)

    if args.head in ("logreg", "both"):
        print("[classify][logreg]", probe(Xtr, ytr, Xev, yev))
    if args.head in ("mlp", "both"):
        print("[classify][mlp]   ", mlp_probe(
            Xtr, ytr, Xev, yev, hidden=args.hidden, epochs=args.epochs,
            lr=args.lr, weight_decay=args.weight_decay, seed=args.seed))


if __name__ == "__main__":
    main()

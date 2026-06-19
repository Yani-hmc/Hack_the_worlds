# Phase 3a — visualize frozen recording-level features (UMAP + t-SNE).
# Run on Dalia (GPU + dataset required):
#   python -m examples.eeg.viz --ckpt <.../latest.pth.tar> --out viz_out
# Produces viz_out/umap.png and viz_out/tsne.png (2D scatter, colored
# normal/abnormal) from the FROZEN encoder embeddings of the eval split (and
# train, faded). umap-learn is optional: if missing, UMAP is skipped gracefully.
"""Embed held-out EEG recordings with the frozen encoder and plot UMAP/t-SNE."""
import argparse
import os

import numpy as np
import torch
from omegaconf import OmegaConf

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt

from examples.eeg.main import build_encoder
from examples.eeg.eval import extract_features


def _scatter(ax, emb, y, title, alpha=0.8):
    for label, name, color in [(0, "normal", "#2b8cbe"), (1, "abnormal", "#e34a33")]:
        m = y == label
        ax.scatter(emb[m, 0], emb[m, 1], s=10, alpha=alpha, c=color,
                   label=f"{name} (n={int(m.sum())})", edgecolors="none")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="viz_out")
    ap.add_argument("--split", default="eval", choices=["train", "eval"],
                    help="which patient-disjoint split to embed/plot")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    encoder = build_encoder(cfg.model).to(device)
    encoder.load_state_dict(state["encoder"]); encoder.eval()

    print(f"[viz] extracting {args.split} embeddings...", flush=True)
    X, y = extract_features(encoder, args.split, device)
    print(f"[viz] {X.shape[0]} recordings, dim={X.shape[1]}, "
          f"frac_abnormal={float(np.mean(y)):.3f}", flush=True)

    # standardize features before 2D embedding (stabilizes both methods)
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)

    # ---- t-SNE (always available via scikit-learn) ----
    from sklearn.manifold import TSNE
    perplexity = min(30, max(5, (len(y) - 1) // 3))
    print(f"[viz] t-SNE (perplexity={perplexity})...", flush=True)
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=args.seed)
    emb_tsne = tsne.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter(ax, emb_tsne, y, f"t-SNE — {args.split} (frozen features)")
    fig.tight_layout(); p = os.path.join(args.out, "tsne.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[viz] wrote {p}", flush=True)

    # ---- UMAP (optional dependency) ----
    try:
        import umap  # umap-learn
    except ImportError:
        print("[viz] umap-learn not installed; skipping UMAP "
              "(pip install umap-learn to enable).", flush=True)
        return
    n_neighbors = min(15, max(2, len(y) - 1))
    print(f"[viz] UMAP (n_neighbors={n_neighbors})...", flush=True)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=0.1, random_state=args.seed)
    emb_umap = reducer.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter(ax, emb_umap, y, f"UMAP — {args.split} (frozen features)")
    fig.tight_layout(); p = os.path.join(args.out, "umap.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[viz] wrote {p}", flush=True)


if __name__ == "__main__":
    main()

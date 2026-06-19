# Phase 5 — sweep the auxiliary spectral-loss coefficients.
# Run on Dalia (GPU + dataset required):
#   python -m examples.eeg.sweep_spectral --fname examples/eeg/cfgs/sweep_spectral.yaml
# For each (spectral_coeff, fft_consistency_coeff) grid point it pretrains an SSL
# model then runs the patient-disjoint probe, and prints/saves a results table
# (sorted by eval AUROC). Single-GPU, sequential.
"""Grid sweep over the Phase-5 spectral auxiliary-loss coefficients."""
import argparse
import itertools
import json
import os

from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig
from examples.eeg.main import run as ssl_run, build_encoder
from examples.eeg.eval import extract_features, probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fname", default="examples/eeg/cfgs/sweep_spectral.yaml")
    args = ap.parse_args()

    scfg = OmegaConf.load(args.fname)
    base = OmegaConf.load(scfg.base_cfg)
    if "epochs" in scfg and scfg.epochs is not None:
        base.optim.epochs = int(scfg.epochs)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grid = list(itertools.product(list(scfg.spectral_coeff),
                                  list(scfg.fft_consistency_coeff)))
    os.makedirs(scfg.ckpt_root, exist_ok=True)
    results = []
    for sc, fc in grid:
        tag = f"spec{sc}_fft{fc}".replace(".", "p")
        ckpt_dir = os.path.join(scfg.ckpt_root, tag)
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
        cfg.model.spectral_coeff = float(sc)
        cfg.model.fft_consistency_coeff = float(fc)
        cfg.meta.ckpt_dir = ckpt_dir
        print(f"\n=== sweep point {tag}: spectral={sc} fft_consistency={fc} ===",
              flush=True)
        ssl_run(cfg=cfg, folder=ckpt_dir)

        # reload the just-trained encoder and probe it
        ckpt = os.path.join(ckpt_dir, "latest.pth.tar")
        state = torch.load(ckpt, map_location=device, weights_only=False)
        rcfg = OmegaConf.create(state["cfg"])
        enc = build_encoder(rcfg.model).to(device)
        enc.load_state_dict(state["encoder"]); enc.eval()
        Xtr, ytr = extract_features(enc, "train", device)
        Xev, yev = extract_features(enc, "eval", device)
        metrics = probe(Xtr, ytr, Xev, yev)
        row = {"spectral_coeff": float(sc), "fft_consistency_coeff": float(fc),
               **metrics}
        print(f"[sweep] {tag} -> {metrics}", flush=True)
        results.append(row)

    results.sort(key=lambda r: r.get("auroc", 0.0), reverse=True)
    out = os.path.join(scfg.ckpt_root, "sweep_results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print("\n[sweep] sorted by AUROC:")
    for r in results:
        print(f"  spec={r['spectral_coeff']:<5} fft={r['fft_consistency_coeff']:<5} "
              f"auroc={r.get('auroc')} bacc={r.get('balanced_accuracy')}", flush=True)
    print(f"[sweep] results -> {out}")


if __name__ == "__main__":
    main()

# Phase 4 — end-to-end fine-tuning of the SSL encoder for TUAB abnormality.
# Run on Dalia (GPU + dataset required):
#   python -m examples.eeg.finetune --fname examples/eeg/cfgs/finetune.yaml \
#       --init <.../latest.pth.tar>
#   # from scratch (supervised baseline): drop --init
# Loads the pretrained encoder (--init), attaches a classification head, trains the
# WHOLE model END-TO-END on TUAB train patients, and evaluates patient-disjoint on
# eval. Reports accuracy/balanced-acc/precision/recall/F1/AUROC at recording level.
"""End-to-end fine-tuning of the EEG SSL encoder + a classification head."""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.main import build_encoder
from examples.eeg.eval import _metrics


class EEGClassifier(nn.Module):
    """Frozen-or-finetuned encoder + (optional MLP) classification head.

    forward takes windows [M, C, T] -> per-window logits [M, 2]. Recording-level
    scores are obtained by aggregating the M=n_windows logits of each recording.
    """

    def __init__(self, encoder, out_dim, head_hidden=256, head_dropout=0.3):
        super().__init__()
        self.encoder = encoder
        if head_hidden and head_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(out_dim, head_hidden), nn.BatchNorm1d(head_hidden),
                nn.ReLU(), nn.Dropout(head_dropout), nn.Linear(head_hidden, 2),
            )
        else:
            self.head = nn.Linear(out_dim, 2)

    def forward(self, x):                 # x: [M, C, T]
        return self.head(self.encoder.represent(x))


def _make_probe_loader(dcfg, split, shuffle):
    cfg = EEGConfig(**{**dcfg, "split": split, "mode": "probe"})
    ds = EEGDataset(cfg)
    return torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0)


def _agg_logits(logits, B, N, mode):
    """[B*N, 2] per-window logits -> [B, 2] recording-level logits."""
    lg = logits.reshape(B, N, 2)
    if mode == "logit_mean":
        return lg.mean(dim=1)
    # mean of softmax probabilities, returned as log-prob-ish logits
    p = torch.softmax(lg, dim=-1).mean(dim=1)
    return torch.log(p.clamp_min(1e-8))


@torch.no_grad()
def evaluate(model, loader, device, agg):
    model.eval()
    proba, pred, ys = [], [], []
    for wins, labels, ok in loader:
        B, N = wins.shape[0], wins.shape[1]
        flat = wins.reshape(B * N, *wins.shape[2:]).to(device, non_blocking=True)
        rec = _agg_logits(model(flat), B, N, agg)
        pr = torch.softmax(rec, dim=-1)[:, 1].cpu().numpy()
        pd = rec.argmax(dim=-1).cpu().numpy()
        for k in range(B):
            if bool(ok[k]):
                proba.append(pr[k]); pred.append(pd[k]); ys.append(int(labels[k]))
    return _metrics(np.array(ys), np.array(pred), np.array(proba), ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fname", default="examples/eeg/cfgs/finetune.yaml")
    ap.add_argument("--init", default=None, help="SSL checkpoint to init encoder")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.fname)
    if args.init is not None:
        cfg.meta.init = args.init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    dcfg = OmegaConf.to_container(cfg.data, resolve=True)

    encoder = build_encoder(cfg.model).to(device)
    if cfg.meta.get("init", None):
        print(f"[ft] loading SSL encoder init: {cfg.meta.init}", flush=True)
        state = torch.load(cfg.meta.init, map_location=device, weights_only=False)
        encoder.load_state_dict(state["encoder"])
    else:
        print("[ft] no --init: training encoder from scratch (supervised baseline)",
              flush=True)

    model = EEGClassifier(
        encoder, out_dim=cfg.model.out_dim,
        head_hidden=cfg.model.get("head_hidden", 256),
        head_dropout=cfg.model.get("head_dropout", 0.3),
    ).to(device)

    # discriminative LRs: encoder fine-tunes slower than the fresh head
    enc_lr = cfg.optim.lr * cfg.optim.get("encoder_lr_mult", 0.1)
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": enc_lr},
        {"params": model.head.parameters(), "lr": cfg.optim.lr},
    ], weight_decay=cfg.optim.weight_decay)
    agg = cfg.optim.get("agg", "mean")

    train_loader = _make_probe_loader(dcfg, "train", shuffle=True)
    eval_loader = _make_probe_loader(dcfg, "eval", shuffle=False)

    # class-balanced loss from train label frequencies
    print("[ft] scanning train labels for class balance...", flush=True)
    ytr = [lab for _, lab in EEGDataset(
        EEGConfig(**{**dcfg, "split": "train", "mode": "probe"})).items]
    counts = np.bincount(ytr, minlength=2).astype(np.float64)
    w = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(w, dtype=torch.float32, device=device))

    os.makedirs(cfg.meta.ckpt_dir, exist_ok=True)
    for epoch in range(cfg.optim.epochs):
        model.train()
        for wins, labels, ok in train_loader:
            B, N = wins.shape[0], wins.shape[1]
            flat = wins.reshape(B * N, *wins.shape[2:]).to(device, non_blocking=True)
            # per-window targets = recording label repeated N times
            tgt = labels.to(device).repeat_interleave(N)
            okm = ok.to(device).repeat_interleave(N).bool()
            logits = model(flat)
            if okm.any():
                loss = crit(logits[okm], tgt[okm])
            else:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
        m = evaluate(model, eval_loader, device, agg)
        print(f"[ft] epoch {epoch} loss={loss.item():.4f} eval={m}", flush=True)
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(cfg.meta.ckpt_dir, "latest.pth.tar"))
    print(f"[ft] done -> {cfg.meta.ckpt_dir}/latest.pth.tar")


if __name__ == "__main__":
    main()

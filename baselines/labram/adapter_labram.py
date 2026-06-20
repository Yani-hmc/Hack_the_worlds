"""LaBraM-Base adapter — run the official ICLR-2024 SOTA EEG foundation model
(github.com/935963004/LaBraM) on OUR TUAB_PREPROCESSED pipeline, starting from
the authors' pretrained checkpoint that ships in the repo
(`external/LaBraM/checkpoints/labram-base.pth`, 5.8M-param NeuralTransformer).

This mirrors `baselines/biot/adapter_biot.py`: the MODEL is the upstream one
(imported from the cloned repo's `modeling_finetune.py`, weights = the authors'
pretrained checkpoint), only the DATA (our patient-disjoint TUAB windows) and the
EVAL protocol (per-window + per-recording) are ours.

LaBraM input contract (from engine_for_finetuning.py):
    EEG [B, C, T=2000]  ->  /100 (µV scaling)  ->  rearrange 'B N (A T) -> B N A T', T=200
    model(x, input_chans)  where input_chans = [0(cls)] + [standard_1020.index(ch)+1 ...]

TWO honest deviations from LaBraM's official TUAB recipe, both documented so the
number is interpreted correctly:
  1. SCALE. LaBraM divides raw-µV by 100; OUR windows are already per-channel
     z-scored (std≈1). We therefore do NOT divide by 100 — we feed the z-scored
     window directly. LaBraM's patch-embed (TemporalConv) starts with GroupNorm,
     which re-normalises per-patch, so a constant input-scale factor is absorbed;
     end-to-end fine-tuning adapts the rest. (A 100× error -- z-score AND /100 --
     would zero out the signal, hence we drop the /100.)
  2. CHANNEL ORDER. Our pipeline doesn't carry channel names; we assume the
     canonical TUAB "01_tcp_ar" 19-montage order (below). If the preprocessed
     order differs, the per-channel position embeddings are mis-assigned (FT
     partially compensates). The number is a faithful-as-possible reproduction,
     not a bit-exact rerun of LaBraM's TUAB pipeline.

RUN ON DALIA (compute node), with the LaBraM repo on PYTHONPATH and timm==0.4.12:
    PYTHONPATH=$WORK/external/LaBraM:$PYTHONPATH \
      uv run python -m baselines.labram.adapter_labram \
        --ckpt $WORK/external/LaBraM/checkpoints/labram-base.pth \
        --mode finetune --epochs 20 --batch-size 64
"""
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset

# Canonical TUAB 01_tcp_ar 19-channel montage order (no A1/A2/T1/T2).
CH_NAMES_19 = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
               'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']

# Copied VERBATIM from LaBraM utils.py (importing utils pulls in deepspeed, so we
# vendor the two trivial pieces). The index of each name = its position embedding.
standard_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]


def get_input_chans(ch_names):
    input_chans = [0]  # cls token
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name) + 1)
    return input_chans


def build_labram(ckpt_path, num_classes=2, drop_path=0.1):
    """Build labram_base from upstream modeling_finetune.py + load pretrained ckpt.

    Kwargs replicate run_class_finetuning.py::create_model() with the TUAB finetune
    flags from the repo README (`--abs_pos_emb --disable_rel_pos_bias`): absolute
    spatial position embedding ON (indexed by input_chans = our channels), relative
    position bias OFF, layer-scale 0.1, qkv_bias ON. These must match the values the
    checkpoint was trained with or the pretrained weights won't align.
    """
    import modeling_finetune  # noqa: F401 — registers the builders, needs timm
    model = modeling_finetune.labram_base_patch200_200(
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=drop_path,
        attn_drop_rate=0.0,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=True,
    )
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = None
        if isinstance(ckpt, dict):
            for k in ("model", "module"):
                if k in ckpt:
                    state = ckpt[k]
                    break
        state = state if state is not None else ckpt
        # labram-base.pth is a DINO-style pretrain ckpt: the encoder lives under a
        # "student." prefix (teacher + decoder + logit_scale are the other top keys).
        # run_class_finetuning strips exactly this via --model_filter_name (key[8:]).
        if any(k.startswith("student.") for k in state):
            state = {k[len("student."):]: v for k, v in state.items()
                     if k.startswith("student.")}
        # classification head is task-specific -> drop, will be fresh-initialised
        for k in ("head.weight", "head.bias"):
            state.pop(k, None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[labram] loaded {ckpt_path}", flush=True)
        print(f"[labram]   missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        print(f"[labram]   e.g. missing={missing[:3]} unexpected={unexpected[:3]}", flush=True)
    return model


def labram_forward(model, x, input_chans):
    """x: [B, C, 2000] z-scored -> [B, C, 10, 200] -> logits [B, num_classes]."""
    B, C, T = x.shape
    x = x.reshape(B, C, T // 200, 200)
    return model(x, input_chans=input_chans)


# --------------------------------------------------------------------------- #
# Window-level view — shared helper supports n_windows=-1 (ALL non-overlapping
# windows per recording, literature protocol).
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
def evaluate(model, loader, device, input_chans):
    model.eval()
    probs, labels, recs, oks = [], [], [], []
    for x, y, rec, ok in loader:
        x = x.to(device)
        p = F.softmax(labram_forward(model, x, input_chans), dim=1)[:, 1].cpu().numpy()
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


def _lr_at(epoch, base, warmup, total):
    if epoch < warmup:
        return base * (epoch + 1) / max(1, warmup)
    p = (epoch - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="path to labram-base.pth")
    ap.add_argument("--mode", choices=["finetune", "frozen"], default="finetune")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--n-channels", type=int, default=19)
    ap.add_argument("--n-windows", type=int, default=-1,
                    help="-1 = ALL non-overlapping (literature protocol); 16 = legacy")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="CPU build+one-forward sanity check, no training")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ch_names = CH_NAMES_19[:args.n_channels]
    input_chans = torch.tensor(get_input_chans(ch_names), dtype=torch.long, device=device)
    print(f"[labram] device={device} mode={args.mode} n_channels={args.n_channels}", flush=True)
    print(f"[labram] input_chans={input_chans.tolist()}", flush=True)

    model = build_labram(args.ckpt).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[labram] params={n_par/1e6:.2f}M", flush=True)

    if args.smoke:
        x = torch.randn(4, args.n_channels, 2000, device=device)
        out = labram_forward(model, x, input_chans)
        print(f"[labram] SMOKE ok: forward {tuple(x.shape)} -> {tuple(out.shape)}", flush=True)
        return

    if args.mode == "frozen":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.head.parameters():
            p.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.parameters()

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

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        lr = _lr_at(epoch, args.lr, args.warmup_epochs, args.epochs) \
            if args.mode == "finetune" else args.lr
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        running = 0.0
        for x, y, _, ok in train_loader:
            ok = ok.bool()
            if ok.sum() == 0:
                continue
            x = x[ok].to(device); y = y[ok].to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(labram_forward(model, x, input_chans), y)
            loss.backward(); opt.step()
            running = loss.item()
        print(f"[labram] epoch {epoch} lr={lr:.2e} loss={running:.4f}", flush=True)

    win, rec = evaluate(model, eval_loader, device, input_chans)
    print(f"[labram] EVAL ({args.mode}, patient-disjoint)")
    print("  per-window     :", win)
    print("  per-recording  :", rec, "  <-- compare to JEPA frozen probe / BIOT")
    print(f"=== DONE LaBraM {args.mode} ===")


if __name__ == "__main__":
    main()

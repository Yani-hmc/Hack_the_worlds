# Multi-corpus SSL pretraining (the foundation-model recipe, à la LaBraM/BIOT):
# pretrain the JEPA on ALL available unlabeled TUH EEG, then probe on TUAB (the golden
# benchmark the papers use) -> directly comparable results.
#
# Pretrain pool (19 ch @ 200 Hz; TUAR read as first-19 of 23): TUAB-train + TUEP + TUSZ + TUAR
# (~13k recordings). TUAB-EVAL patient-ids are EXCLUDED from pretraining (no leakage).
#
# Run (Dalia):
#   python -m examples.eeg.pretrain_multicorpus --fname examples/eeg/cfgs/train.yaml \
#       meta.ckpt_dir=$WORK/checkpoints/eeg/multicorpus model.ssl_loss=sigreg \
#       data.aug_exact_corruption=true data.epoch_size=40000 optim.epochs=30
#   python -m examples.eeg.eval --ckpt $WORK/checkpoints/eeg/multicorpus/latest.pth.tar --level both
import glob
import os
import sys

import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.main import build_encoder, build_ssl

NEURO = "/lustre/work/pdl17890/udl806719/datasets/Neuro"
PRETRAIN_ROOTS = [
    NEURO + "/TUAB-TUEV/TUAB_PREPROCESSED/train",   # 2717  (abnormality)
    NEURO + "/TUAR-TUEP/TUEP_PREPROCESSED",         # 2795  (epilepsy)
    NEURO + "/TUSZ_PREPROCESSED/edf",               # 7511  (seizure)
    NEURO + "/TUAR-TUEP/TUAR_PREPROCESSED/edf",     # 290   (artifact, 23ch -> first 19)
]
TUAB_EVAL = NEURO + "/TUAB-TUEV/TUAB_PREPROCESSED/eval"


def _pid(path):
    """TUH patient id = first 8 chars of the filename (shared across sub-corpora)."""
    return os.path.basename(path).split("_")[0][:8]


class MultiCorpusEEG(EEGDataset):
    """SSL dataset over MANY corpora. Reuses the TUAB loader's EDF partial-read +
    two-view augmentation; only the file list changes (multi-corpus, leakage-filtered)."""

    def __init__(self, cfg, roots, exclude_pids):
        assert cfg.mode == "ssl"
        super().__init__(cfg)                       # builds pyedflib + (TUAB) self.files
        files = []
        for r in roots:
            files += glob.glob(os.path.join(r, "**", "*.edf"), recursive=True)
        n0 = len(files)
        files = [f for f in files if _pid(f) not in exclude_pids]   # no TUAB-eval leakage
        self.files = sorted(files)
        print(f"[multicorpus] {len(self.files)} pretrain recordings "
              f"({n0 - len(self.files)} excluded for TUAB-eval patient overlap)", flush=True)


def run(fname, **ov):
    cfg = OmegaConf.load(fname)
    if ov:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in ov.items()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    dcfg = EEGConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dcfg.mode = "ssl"
    exclude = set(_pid(p) for p in
                  glob.glob(os.path.join(TUAB_EVAL, "**", "*.edf"), recursive=True))
    print(f"[multicorpus] excluding {len(exclude)} TUAB-eval patient ids", flush=True)
    ds = MultiCorpusEEG(dcfg, PRETRAIN_ROOTS, exclude)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=dcfg.batch_size, shuffle=True, num_workers=dcfg.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=dcfg.num_workers > 0)

    encoder = build_encoder(cfg.model).to(device)
    ssl = build_ssl(encoder, cfg.model).to(device)
    opt = torch.optim.AdamW(ssl.parameters(), lr=cfg.optim.lr,
                            weight_decay=cfg.optim.weight_decay)

    os.makedirs(cfg.meta.ckpt_dir, exist_ok=True)
    for epoch in range(cfg.optim.epochs):
        ssl.train()
        loss = None
        for batch in loader:
            batch = [b.to(device) for b in batch]
            opt.zero_grad(set_to_none=True)
            loss, logs = ssl.compute_loss(batch)
            loss.backward(); opt.step()
        print(f"[pretrain] epoch {epoch} loss={loss.item():.4f} {logs}", flush=True)
        torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(cfg.meta.ckpt_dir, "latest.pth.tar"))
    print(f"[pretrain] done -> {cfg.meta.ckpt_dir}/latest.pth.tar", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    fname = "examples/eeg/cfgs/train.yaml"
    ov = {}
    i = 0
    while i < len(args):
        if args[i] == "--fname":
            fname = args[i + 1]; i += 2
        elif "=" in args[i]:
            k, v = args[i].split("=", 1); ov[k] = v; i += 1
        else:
            i += 1
    run(fname, **ov)

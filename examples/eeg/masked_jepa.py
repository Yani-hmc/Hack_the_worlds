# Masked-prediction JEPA for EEG (the faithful JEPA objective, à la I-JEPA / V-JEPA / LaBraM).
# Run (on Dalia):
#   python -m examples.eeg.main --fname examples/eeg/cfgs/train.yaml model.ssl_type=masked
"""Masked latent-prediction JEPA over EEG windows.

Instead of two-view invariance (VICReg), this is the predictive JEPA energy:
  * encode a window into a latent SEQUENCE  z = encoder.frames(x) -> [B, L, D]
  * an EMA *target* encoder produces the prediction targets (stop-gradient)
  * randomly MASK a fraction of the L latent frames; a small Transformer predictor
    fills in the masked frames from the visible (context) frames + mask tokens
  * loss = smooth-L1 between predicted and target representations on the MASKED
    frames only, plus a variance+covariance anti-collapse term on the online frames.

This mirrors how LaBraM / V-JEPA learn (masked modeling in representation space),
which is the regime that drives SOTA — as opposed to the invariance form.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from eb_jepa.losses import CovarianceLoss, HingeStdLoss


def _sinusoidal(L, D, device):
    """Standard sinusoidal positional encoding -> [1, L, D] (handles any L)."""
    pos = torch.arange(L, device=device).unsqueeze(1).float()
    i = torch.arange(0, D, 2, device=device).float()
    div = torch.exp(-math.log(10000.0) * i / D)
    pe = torch.zeros(L, D, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class MaskedJEPA(nn.Module):
    """Masked latent-prediction JEPA with an EMA target encoder."""

    def __init__(self, encoder, cfg):
        super().__init__()
        self.encoder = encoder                              # online (gradient) encoder
        self.target_encoder = copy.deepcopy(encoder)        # EMA target (no gradient)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.ema = float(getattr(cfg, "ema", 0.996))
        self.mask_ratio = float(getattr(cfg, "mask_ratio", 0.5))
        self.std_coeff = float(getattr(cfg, "std_coeff", 25.0))
        self.cov_coeff = float(getattr(cfg, "cov_coeff", 1.0))

        D = encoder.out_dim
        depth = int(getattr(cfg, "pred_depth", 2))
        heads = int(getattr(cfg, "pred_heads", 4))
        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=heads, dim_feedforward=2 * D,
            batch_first=True, activation="gelu", dropout=0.0,
        )
        self.predictor = nn.TransformerEncoder(layer, num_layers=depth)
        self.proj = nn.Linear(D, D)                          # predictor head
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.normal_(self.mask_token, std=0.02)
        self.std_loss = HingeStdLoss(std_margin=1.0)
        self.cov_loss = CovarianceLoss()

    @torch.no_grad()
    def _ema_update(self):
        for po, pt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            pt.mul_(self.ema).add_(po.detach(), alpha=1.0 - self.ema)
        for bo, bt in zip(self.encoder.buffers(), self.target_encoder.buffers()):
            bt.copy_(bo)

    def compute_loss(self, batch):
        # the ssl loader returns (v1, v2); masked-JEPA needs only one view
        x = batch[0] if isinstance(batch, (list, tuple)) else batch     # [B, 19, T]
        self._ema_update()                                              # EMA before targets

        online = self.encoder.frames(x)                                 # [B, L, D]
        with torch.no_grad():
            target = self.target_encoder.frames(x)                      # [B, L, D] stop-grad
        B, L, D = online.shape
        n_mask = max(1, int(self.mask_ratio * L))

        # per-sample random masked-frame indices
        perm = torch.argsort(torch.rand(B, L, device=online.device), dim=1)
        masked_idx = perm[:, :n_mask]                                   # [B, n_mask]
        mask = torch.zeros(B, L, dtype=torch.bool, device=online.device)
        mask.scatter_(1, masked_idx, True)

        # predictor input: visible frames kept, masked frames replaced by mask token
        ctx = torch.where(mask.unsqueeze(-1), self.mask_token.expand(B, L, D), online)
        ctx = ctx + _sinusoidal(L, D, online.device)
        pred = self.proj(self.predictor(ctx))                          # [B, L, D]

        # prediction loss on MASKED frames only (smooth-L1 in representation space)
        m = mask.unsqueeze(-1).expand_as(pred)
        pred_loss = F.smooth_l1_loss(pred[m], target[m])

        # anti-collapse on the online frames (variance + covariance), flattened [B*L, D]
        flat = online.reshape(B * L, D)
        std_l = self.std_loss(flat)
        cov_l = self.cov_loss(flat)
        loss = pred_loss + self.std_coeff * std_l + self.cov_coeff * cov_l
        logs = {"pred": round(pred_loss.item(), 4),
                "std": round(std_l.item(), 4),
                "cov": round(cov_l.item(), 4)}
        return loss, logs


if __name__ == "__main__":  # tiny CPU self-check, no data needed
    class _Enc(nn.Module):
        out_dim = 32
        def frames(self, x):
            B = x.shape[0]
            return torch.randn(B, 20, 32, requires_grad=True)
    from omegaconf import OmegaConf
    m = MaskedJEPA(_Enc(), OmegaConf.create({"mask_ratio": 0.5}))
    l, d = m.compute_loss((torch.randn(4, 19, 2000), torch.randn(4, 19, 2000)))
    print("masked-jepa self-check ok:", float(l), d)

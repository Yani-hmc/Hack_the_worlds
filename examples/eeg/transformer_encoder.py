# Transformer encoder for EEG windows — the capacity upgrade our ablations point to.
# Drop-in replacement for EEG1DEncoder: same interface (represent / feature_map /
# frames / out_dim), so it works with the existing SSL objectives (VICReg / SIGReg /
# masked) and eval probe unchanged. Select via cfg.model.encoder_type = "transformer".
"""A small ViT-style Transformer over an EEG window [B, C=19, T].

Pipeline: patch-embed the window into L tokens (strided Conv1d over time, mixing the
19 channels into each patch), add sinusoidal positions, run a Transformer encoder,
then mean-pool the tokens -> [B, D]. This mirrors how BIOT/LaBraM operate (attention
over signal patches) at a hackathon-feasible size, directly attacking the capacity
bottleneck a tiny conv encoder can't overcome.
"""
import math

import torch
import torch.nn as nn


def _sinusoidal(L, D, device):
    pos = torch.arange(L, device=device).unsqueeze(1).float()
    i = torch.arange(0, D, 2, device=device).float()
    div = torch.exp(-math.log(10000.0) * i / D)
    pe = torch.zeros(L, D, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class EEGTransformerEncoder(nn.Module):
    """ViT-style encoder. Interface matches EEG1DEncoder."""

    def __init__(self, in_channels=19, out_dim=256, patch=100, depth=4,
                 heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        # patch embed: each `patch` timesteps (×19 ch) -> one D-dim token
        self.patch_embed = nn.Conv1d(in_channels, out_dim,
                                     kernel_size=patch, stride=patch)
        layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=heads, dim_feedforward=mlp_ratio * out_dim,
            batch_first=True, activation="gelu", dropout=dropout)
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim

    def feature_map(self, x):          # [B, C, T] -> [B, D, L]
        t = self.patch_embed(x)        # [B, D, L]
        z = t.transpose(1, 2)          # [B, L, D]
        z = z + _sinusoidal(z.shape[1], z.shape[2], z.device)
        z = self.norm(self.transformer(z))   # [B, L, D]
        return z.transpose(1, 2)       # [B, D, L]

    def represent(self, x):            # [B, C, T] -> [B, D]
        return self.feature_map(x).mean(dim=2)

    def frames(self, x):               # [B, C, T] -> [B, L, D]
        return self.feature_map(x).transpose(1, 2)

    def forward(self, x):
        return self.represent(x)


def build_transformer_encoder(cfg):
    return EEGTransformerEncoder(
        in_channels=cfg.in_channels,
        out_dim=cfg.out_dim,
        patch=int(getattr(cfg, "patch", 100)),
        depth=int(getattr(cfg, "depth", 4)),
        heads=int(getattr(cfg, "heads", 8)),
        mlp_ratio=int(getattr(cfg, "mlp_ratio", 4)),
        dropout=float(getattr(cfg, "tr_dropout", 0.1)),
    )


if __name__ == "__main__":   # CPU shape self-check
    from omegaconf import OmegaConf
    enc = build_transformer_encoder(OmegaConf.create(
        {"in_channels": 19, "out_dim": 256, "patch": 100, "depth": 4}))
    x = torch.randn(4, 19, 2000)
    print("represent:", enc.represent(x).shape, "| feature_map:", enc.feature_map(x).shape,
          "| params(M):", round(sum(p.numel() for p in enc.parameters()) / 1e6, 2))

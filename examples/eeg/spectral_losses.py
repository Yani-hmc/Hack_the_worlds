# Phase 5 — auxiliary spectral losses for the EEG SSL objective.
# This module is imported by examples/eeg/main.py; it is not a runnable script.
# It is exercised end-to-end by the SSL training command:
#   python -m examples.eeg.main --fname examples/eeg/cfgs/train.yaml \
#       model.spectral_coeff=0.1 model.fft_consistency_coeff=0.05
# Self-check (CPU, no data needed) on Dalia or locally:
#   python -m examples.eeg.spectral_losses
"""Multi-scale spectral loss (DDSP) + FFT-magnitude consistency for EEG views.

Reference
---------
Engel, Hantrakul, Gu, Roberts, "DDSP: Differentiable Digital Signal Processing",
ICLR 2020 (arXiv:2001.04643). The multi-scale spectral loss (their Section 4.2 /
Appendix) sums, over several FFT sizes, an L1 distance on the magnitude
spectrogram plus an L1 distance on the *log* magnitude spectrogram:

    L_spec = sum_i [ || S_i - S_i_hat ||_1  +  alpha * || log S_i - log S_i_hat ||_1 ]

where ``i`` ranges over FFT sizes (the paper uses 2048, 1024, 512, 256, 128, 64),
``S_i`` / ``S_i_hat`` are magnitude spectrograms at FFT size ``i`` for the target
and the reconstruction, and ``alpha`` weights the log term (paper uses alpha=1.0).

Adaptation for EEG
------------------
* EEG windows are short ([B, C=19, T=2000] @ 200 Hz), so we drop FFT sizes larger
  than the window and default to ``(256, 128, 64, 32)``.
* There is no decoder/reconstruction in two-view VICReg, so we use this purely as
  a *consistency / invariance* term between the spectrograms of the two corrupted
  views (v1, v2). It pushes the model's two views to share spectral content, which
  is a strong, physically meaningful invariance for EEG (band power is the signal).
* STFT is applied per (batch, channel) over the time axis and averaged over the
  channel and batch dims so the term is scale-stable across configs.
"""
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: torch.Tensor) -> torch.Tensor:
    """Magnitude STFT of a 1D-over-time signal.

    Args:
        x:   [N, T] real signal (N = flattened batch*channels).
        n_fft: FFT size; hop: hop length; win: [n_fft] window on x.device/dtype.
    Returns:
        magnitude spectrogram [N, F, frames].
    """
    spec = torch.stft(
        x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=win, center=True, return_complex=True, pad_mode="reflect",
    )
    return spec.abs()


class MultiScaleSpectralLoss(nn.Module):
    """DDSP multi-scale spectral loss between two [B, C, T] signals.

    L = sum_i [ ||S_i - S'_i||_1 + alpha * ||log S_i - log S'_i||_1 ]
    summed over FFT sizes ``i`` and averaged over the batch/channel/time-frame
    elements (mean reduction) so the magnitude is independent of B, C and T.
    """

    def __init__(self, fft_sizes: Sequence[int] = (256, 128, 64, 32),
                 alpha: float = 1.0, hop_ratio: float = 0.25, eps: float = 1e-5):
        super().__init__()
        self.fft_sizes = tuple(int(n) for n in fft_sizes)
        self.alpha = float(alpha)
        self.hop_ratio = float(hop_ratio)
        self.eps = float(eps)
        # cache Hann windows per fft size (registered so they move with .to(device))
        for n in self.fft_sizes:
            self.register_buffer(f"win_{n}", torch.hann_window(n), persistent=False)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a, b: [B, C, T] -> flatten to [B*C, T]
        B, C, T = a.shape
        af = a.reshape(B * C, T)
        bf = b.reshape(B * C, T)
        loss = af.new_zeros(())
        for n in self.fft_sizes:
            if n > T:
                continue
            hop = max(1, int(n * self.hop_ratio))
            win = getattr(self, f"win_{n}").to(af.dtype)
            Sa = _stft_mag(af, n, hop, win)
            Sb = _stft_mag(bf, n, hop, win)
            lin = F.l1_loss(Sa, Sb)
            log = F.l1_loss(torch.log(Sa + self.eps), torch.log(Sb + self.eps))
            loss = loss + lin + self.alpha * log
        return loss


class FFTMagConsistencyLoss(nn.Module):
    """Simple whole-window FFT magnitude consistency between two [B, C, T] signals.

    Single rFFT over the full window (no framing), L1 on the magnitude spectrum.
    Cheaper and lower-variance than the multi-scale STFT term; captures global band
    power. Averaged over batch/channel/frequency (mean reduction).
    """

    def __init__(self, log: bool = True, eps: float = 1e-5):
        super().__init__()
        self.log = bool(log)
        self.eps = float(eps)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        Ma = torch.fft.rfft(a, dim=-1).abs()
        Mb = torch.fft.rfft(b, dim=-1).abs()
        if self.log:
            Ma = torch.log(Ma + self.eps)
            Mb = torch.log(Mb + self.eps)
        return F.l1_loss(Ma, Mb)


if __name__ == "__main__":
    # CPU smoke test — no dataset needed.
    torch.manual_seed(0)
    x = torch.randn(4, 19, 2000)
    y = x + 0.01 * torch.randn_like(x)
    ms = MultiScaleSpectralLoss()
    fc = FFTMagConsistencyLoss()
    print("multi-scale spectral:", float(ms(x, y)))
    print("fft consistency     :", float(fc(x, y)))
    # gradient sanity
    z = x.clone().requires_grad_(True)
    (ms(z, y) + fc(z, y)).backward()
    print("grad ok:", z.grad is not None and torch.isfinite(z.grad).all().item())

"""EEG dataset — TUH EEG abnormality corpus (TUAB_PREPROCESSED).

Raw EDF recordings (19 channels @ 200 Hz). The corpus is split into
``train`` / ``eval`` patients (patient-disjoint), each with ``normal`` and
``abnormal`` sub-folders.

Two access modes, both PROVIDED here (plumbing):

  * ``mode="ssl"``  — labels are IGNORED. Each ``__getitem__`` reads a random
    10 s window from a random recording and returns TWO independently augmented
    views ``(v1, v2)`` for a two-view (VICReg) invariance objective.
  * ``mode="supervised"`` / ``"probe"`` — one item = one *recording*: N
    evenly-spaced windows ``[N, C, T]`` plus its label (0=normal, 1=abnormal),
    for recording-level feature extraction in ``examples/eeg/eval.py``.

The modelling choices on top of these windows (encoder, SSL objective, probe)
live in ``examples/eeg/`` and are where the ``# TODO``s are.
"""
import glob
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

try:
    import pyedflib
except ImportError:  # surfaced clearly at runtime if the dep is missing
    pyedflib = None


@dataclass
class EEGConfig:
    data_root: str = (
        "/lustre/work/pdl17890/udl806719/datasets/Neuro/TUAB-TUEV/TUAB_PREPROCESSED"
    )
    split: str = "train"           # train | eval (patient-disjoint)
    mode: str = "ssl"              # ssl (two views) | supervised/probe ((windows, label))
    n_channels: int = 19
    sfreq: int = 200               # Hz (TUAB_PREPROCESSED)
    window_sec: float = 10.0       # window length in seconds
    epoch_size: int = 20000        # virtual samples per epoch (random windows, ssl)
    n_windows: int = 16            # evenly-spaced windows per recording (probe mode)
    batch_size: int = 128
    num_workers: int = 8
    # SSL augmentation strengths (per view, in z-scored units)
    aug_noise_std: float = 0.1     # additive Gaussian noise std
    aug_scale_jitter: float = 0.2  # per-channel amplitude scale ~ U(1-j, 1+j)
    aug_chan_drop_p: float = 0.2   # prob a channel is zeroed
    aug_time_mask_frac: float = 0.2  # max fraction of timesteps masked
    # --- Phase 2: EB time-series JEPA "exact" corruption ----------------- #
    # Each view is ~40% corrupted = ~20% time-masking + ~20% outlier injection.
    # When ``aug_outlier_frac > 0`` a fixed fraction of timepoints (per channel)
    # is masked AND a disjoint fraction is replaced by large outliers, instead of
    # the legacy variable-length contiguous mask above. Set aug_outlier_frac=0.0
    # to fall back to the legacy behaviour exactly (default keeps base pipeline).
    aug_exact_corruption: bool = False   # gate: use the exact 20%/20% scheme
    aug_mask_frac: float = 0.2           # fraction of timepoints zero-masked (exact)
    aug_outlier_frac: float = 0.2        # fraction of timepoints replaced by outliers
    aug_outlier_scale: float = 6.0       # outlier magnitude in z-scored sigmas (±k)


def _list_edf(root: str, split: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(root, split, "**", "*.edf"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .edf under {os.path.join(root, split)}")
    return files


def _list_labelled(root: str, split: str):
    """list of (path, label) — label 0=normal, 1=abnormal — for the probe."""
    items = []
    for label, cls in [(0, "normal"), (1, "abnormal")]:
        for p in sorted(glob.glob(os.path.join(root, split, cls, "**", "*.edf"),
                                  recursive=True)):
            items.append((p, label))
    if not items:
        raise FileNotFoundError(
            f"No labelled .edf under {os.path.join(root, split)}/{{normal,abnormal}}")
    return items


def _zscore(x: np.ndarray, axis: int) -> np.ndarray:
    """per-channel z-score (robust to the µV scale)."""
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True) + 1e-6
    return (x - mu) / sd


class EEGDataset(torch.utils.data.Dataset):
    """SSL mode: random windows, two views each. Probe mode: per-recording windows."""

    def __init__(self, cfg: EEGConfig):
        if pyedflib is None:
            raise ImportError(
                "pyedflib is required to read EDF files (pip install pyedflib)")
        self.cfg = cfg
        self.window = int(cfg.window_sec * cfg.sfreq)
        if cfg.mode == "ssl":
            self.files = _list_edf(cfg.data_root, cfg.split)
            self.items = None
        else:  # supervised / probe: one item per recording
            self.files = None
            self.items = _list_labelled(cfg.data_root, cfg.split)
        # one RNG per worker, re-seeded lazily in __getitem__ via torch seed
        self._rng = np.random.default_rng()

    def __len__(self):
        if self.cfg.mode == "ssl":
            return self.cfg.epoch_size
        return len(self.items)

    # ------------------------------------------------------------------ #
    # EDF reading (partial reads — only the windows we need)
    # ------------------------------------------------------------------ #
    def _read_random_window(self) -> Optional[np.ndarray]:
        """Read one [n_channels, window] z-scored window from a random recording."""
        cfg = self.cfg
        for _ in range(8):  # retry on short/unreadable files
            path = self.files[self._rng.integers(len(self.files))]
            try:
                f = pyedflib.EdfReader(path)
            except Exception:
                continue
            try:
                if f.signals_in_file < cfg.n_channels:
                    continue
                nsamp = int(min(f.getNSamples()[:cfg.n_channels]))
                if nsamp <= self.window + 1:
                    continue
                start = int(self._rng.integers(0, nsamp - self.window))
                x = np.empty((cfg.n_channels, self.window), dtype=np.float32)
                for c in range(cfg.n_channels):
                    x[c] = f.readSignal(c, start, self.window)
            finally:
                f._close()
            return _zscore(x, axis=1)
        return None

    def _read_recording_windows(self, path) -> Optional[np.ndarray]:
        """Read N evenly-spaced z-scored windows -> [N, n_channels, window]."""
        cfg, N = self.cfg, self.cfg.n_windows
        try:
            f = pyedflib.EdfReader(path)
        except Exception:
            return None
        try:
            if f.signals_in_file < cfg.n_channels:
                return None
            nsamp = int(min(f.getNSamples()[:cfg.n_channels]))
            if nsamp <= self.window + 1:
                return None
            starts = np.linspace(0, nsamp - self.window, N).astype(int)
            wins = np.empty((N, cfg.n_channels, self.window), dtype=np.float32)
            for c in range(cfg.n_channels):
                for j, s in enumerate(starts):
                    wins[j, c] = f.readSignal(c, int(s), self.window)
        except Exception:
            return None
        finally:
            f._close()
        return _zscore(wins, axis=2)

    # ------------------------------------------------------------------ #
    # SSL augmentation
    # ------------------------------------------------------------------ #
    def _augment(self, x: np.ndarray) -> np.ndarray:
        cfg, rng = self.cfg, self._rng
        x = x.copy()
        # amplitude scale jitter (per channel)
        if cfg.aug_scale_jitter > 0:
            scale = 1.0 + rng.uniform(-cfg.aug_scale_jitter, cfg.aug_scale_jitter,
                                      size=(cfg.n_channels, 1)).astype(np.float32)
            x *= scale
        # additive Gaussian noise
        if cfg.aug_noise_std > 0:
            x += rng.normal(0, cfg.aug_noise_std, size=x.shape).astype(np.float32)
        # per-channel dropout (zeroing)
        if cfg.aug_chan_drop_p > 0:
            mask = (rng.random(cfg.n_channels) > cfg.aug_chan_drop_p).astype(np.float32)
            x *= mask[:, None]
        if getattr(cfg, "aug_exact_corruption", False):
            # ---- Phase 2: exact EB time-series corruption (~40% per view) ----
            # Per channel, pick disjoint random index sets: ~aug_mask_frac of
            # timepoints are zero-masked and ~aug_outlier_frac are replaced by
            # large ±k*sigma outliers (z-scored units). Sampling per channel keeps
            # the two views independently and heterogeneously corrupted.
            T = self.window
            n_mask = int(round(cfg.aug_mask_frac * T))
            n_out = int(round(cfg.aug_outlier_frac * T))
            n_out = min(n_out, max(T - n_mask, 0))  # keep the two sets disjoint
            if n_mask > 0 or n_out > 0:
                for c in range(cfg.n_channels):
                    if n_mask + n_out > T:
                        break
                    idx = rng.permutation(T)
                    if n_mask > 0:
                        x[c, idx[:n_mask]] = 0.0
                    if n_out > 0:
                        oi = idx[n_mask:n_mask + n_out]
                        signs = rng.choice(
                            np.array([-1.0, 1.0], dtype=np.float32), size=n_out)
                        x[c, oi] = (cfg.aug_outlier_scale * signs).astype(np.float32)
        else:
            # legacy behaviour: zero a single random contiguous span
            if cfg.aug_time_mask_frac > 0:
                mlen = int(rng.uniform(0, cfg.aug_time_mask_frac) * self.window)
                if mlen > 0:
                    s = int(rng.integers(0, self.window - mlen))
                    x[:, s:s + mlen] = 0.0
        return x

    def __getitem__(self, i):
        # re-seed per call so workers diverge (torch sets a per-worker base seed)
        self._rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
        if self.cfg.mode == "ssl":
            x = self._read_random_window()
            if x is None:  # fallback: zeros (rare)
                x = np.zeros((self.cfg.n_channels, self.window), dtype=np.float32)
            v1 = torch.from_numpy(self._augment(x))
            v2 = torch.from_numpy(self._augment(x))
            return v1, v2
        # supervised / probe: one recording -> [N, C, T] + label
        path, label = self.items[i]
        w = self._read_recording_windows(path)
        ok = w is not None
        if not ok:
            w = np.zeros((self.cfg.n_windows, self.cfg.n_channels, self.window),
                         dtype=np.float32)
        return torch.from_numpy(w), int(label), ok


def make_loader(cfg: EEGConfig, shuffle=None):
    ds = EEGDataset(cfg)
    is_train = cfg.mode == "ssl" and cfg.split == "train"
    return torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size,
        shuffle=is_train if shuffle is None else shuffle,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=is_train,
        persistent_workers=cfg.num_workers > 0)

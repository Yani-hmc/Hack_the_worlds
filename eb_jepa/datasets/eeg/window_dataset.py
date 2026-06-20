"""Shared per-window labelled dataset used by every supervised adapter.

Two modes (selected by `n_windows`):
- `n_windows=16` (legacy): exactly 16 evenly-spaced windows per recording.
  Total items = len(recs) * 16. The whole pipeline assumes this fixed N.
- `n_windows=-1` (literature-canonical): ALL non-overlapping windows per
  recording (~175 per ~20-min recording → ~46k eval items, ~470k train items).
  Variable N per recording; built via a one-pass EDF-metadata scan at init.

Both modes return `(window, label, rec_idx, ok_flag)` per item — the recording
index lets the eval aggregate per-window probs back to recording level.

The all-windows mode caches the last-read recording's full window stack so that
consecutive same-`rec_i` accesses don't re-read the EDF. With sequential eval
this is ~one read per recording; with shuffled training, cache hit is low and
you pay the full ~N-window EDF read penalty per access. Use more dataloader
workers to compensate.
"""
import os
from typing import Optional

import numpy as np
import torch

try:
    import pyedflib
except ImportError:
    pyedflib = None

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset


class WindowDataset(torch.utils.data.Dataset):
    """Flattens probe-mode recordings into per-window labelled items.

    Args:
        split: "train" or "eval".
        n_windows: 16 = legacy evenly-spaced; -1 = all non-overlapping.
        n_channels: how many channels to feed to the model (default 19).
    """

    def __init__(self, split: str, n_windows: int = 16, n_channels: int = 19):
        if pyedflib is None and n_windows < 0:
            raise ImportError("pyedflib required for n_windows=-1 (all windows)")
        self.cfg = EEGConfig(
            split=split, mode="probe", n_windows=n_windows, n_channels=n_channels
        )
        self.base = EEGDataset(self.cfg)
        self.n_windows = n_windows
        self.n_channels = n_channels

        if n_windows < 0:
            # ALL non-overlapping windows: variable count per recording.
            # Pre-scan EDF metadata for the count, build a flat (rec_i, win_i)
            # index. Recordings whose EDF can't be read get 0 windows (dropped).
            self.index = []
            window_samples = int(self.cfg.sfreq * self.cfg.window_sec)
            for rec_i, (path, _) in enumerate(self.base.items):
                n_i = self._n_windows_for(path, window_samples)
                for win_i in range(n_i):
                    self.index.append((rec_i, win_i))
            # Cache: the last fully-read recording's window stack
            self._cache_rec = -1
            self._cache_data = None
            print(
                f"[WindowDataset n_windows=ALL split={split}] "
                f"{len(self.index)} windows from {len(self.base.items)} recordings "
                f"(~{len(self.index)/max(1, len(self.base.items)):.1f} per rec avg)",
                flush=True,
            )
        else:
            self.index = None
            print(
                f"[WindowDataset n_windows={n_windows} split={split}] "
                f"{len(self.base.items) * n_windows} windows from "
                f"{len(self.base.items)} recordings (fixed N per rec)",
                flush=True,
            )

    @staticmethod
    def _n_windows_for(path: str, window_samples: int) -> int:
        """Cheap EDF metadata read: how many non-overlapping windows fit."""
        try:
            f = pyedflib.EdfReader(path)
            nsamp = int(min(f.getNSamples()[:19]))
            f._close()
            return max(0, nsamp // window_samples)
        except Exception:
            return 0

    def __len__(self):
        if self.n_windows < 0:
            return len(self.index)
        return len(self.base.items) * self.n_windows

    def __getitem__(self, idx):
        if self.n_windows < 0:
            rec_i, win_i = self.index[idx]
        else:
            rec_i, win_i = divmod(idx, self.n_windows)

        # Cache the last-read recording's full window stack
        if self.n_windows < 0:
            if rec_i != self._cache_rec:
                self._cache_data = self.base[rec_i]
                self._cache_rec = rec_i
            wins, label, ok = self._cache_data
        else:
            wins, label, ok = self.base[rec_i]

        if not ok or wins is None or win_i >= wins.shape[0]:
            # Fallback: zeros + ok=0 so the eval can filter
            x = torch.zeros(self.n_channels, int(self.cfg.sfreq * self.cfg.window_sec))
            return x, int(label), int(rec_i), 0
        return wins[win_i], int(label), int(rec_i), int(bool(ok))

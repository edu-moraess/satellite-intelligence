"""Preprocessing transforms for satellite imagery (training / inference)."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def normalize_percentile(
    image: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """Per-band percentile stretch to [0, 1]."""
    out = image.astype(np.float32).copy()
    if out.ndim == 2:
        out = out[np.newaxis, ...]
    for c in range(out.shape[0]):
        band = out[c]
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, (p_low, p_high))
        if hi > lo:
            out[c] = np.clip((band - lo) / (hi - lo), 0, 1)
        else:
            out[c] = 0.0
    return np.nan_to_num(out, nan=0.0)


def to_chw(image: np.ndarray) -> np.ndarray:
    """Ensure channel-first layout."""
    if image.ndim == 3 and image.shape[-1] in (3, 4) and image.shape[0] not in (3, 4):
        return np.transpose(image, (2, 0, 1))
    return image

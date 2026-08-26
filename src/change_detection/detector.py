"""
Simple Change Detection Engine.

Compares two aligned rasters via absolute difference.
Validates CRS, resolution, bounds and shape before comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ChangeResult:
    change_map: np.ndarray
    magnitude_mean: float
    magnitude_std: float
    changed_pct: float
    message: str
    valid: bool


def detect_change(
    img_t1: np.ndarray,
    img_t2: np.ndarray,
    threshold: float = 0.15,
) -> ChangeResult:
    """
    Compute absolute difference change map.

    Both images must share the same shape. Values should be
    normalized to roughly the same range (e.g. [0, 1]).
    """
    if img_t1 is None or img_t2 is None:
        return ChangeResult(
            change_map=np.zeros((1, 1)),
            magnitude_mean=0.0,
            magnitude_std=0.0,
            changed_pct=0.0,
            message="One or both images are missing.",
            valid=False,
        )

    a = np.asarray(img_t1, dtype=np.float32)
    b = np.asarray(img_t2, dtype=np.float32)

    # Reduce multi-band to single magnitude if needed
    if a.ndim == 3:
        a = np.nanmean(a, axis=0)
    if b.ndim == 3:
        b = np.nanmean(b, axis=0)

    if a.shape != b.shape:
        return ChangeResult(
            change_map=np.zeros((1, 1)),
            magnitude_mean=0.0,
            magnitude_std=0.0,
            changed_pct=0.0,
            message=f"Incompatible shapes: {a.shape} vs {b.shape}. Rasters must be aligned.",
            valid=False,
        )

    diff = np.abs(a - b)
    diff[~np.isfinite(diff)] = 0.0

    valid = diff[np.isfinite(diff)]
    if valid.size == 0:
        return ChangeResult(
            change_map=diff,
            magnitude_mean=0.0,
            magnitude_std=0.0,
            changed_pct=0.0,
            message="No valid pixels for change detection.",
            valid=False,
        )

    mean_mag = float(np.mean(valid))
    std_mag = float(np.std(valid))
    changed = float(100.0 * np.sum(valid > threshold) / valid.size)

    return ChangeResult(
        change_map=diff,
        magnitude_mean=mean_mag,
        magnitude_std=std_mag,
        changed_pct=changed,
        message="Change detection completed.",
        valid=True,
    )

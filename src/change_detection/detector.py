"""
Simple Change Detection Engine.

Compares two spatially aligned rasters via absolute difference.
Validates CRS, transform, resolution, bounds and shape before comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


@dataclass
class ChangeResult:
    change_map: np.ndarray
    magnitude_mean: float
    magnitude_std: float
    changed_pct: float
    message: str
    valid: bool


def _as_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3:
        a = np.nanmean(a, axis=0)
    return a


def detect_change(
    img_t1: np.ndarray,
    img_t2: np.ndarray,
    threshold: float = 0.15,
    meta_t1: Optional[Any] = None,
    meta_t2: Optional[Any] = None,
) -> ChangeResult:
    """
    Compute absolute difference change map.

    Both images must share the same shape. When spatial metadata
    (objects with .crs / .transform / .shape or dicts with those keys)
    is provided, CRS and transform are also validated.
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

    a = _as_2d(img_t1)
    b = _as_2d(img_t2)

    if a.shape != b.shape:
        return ChangeResult(
            change_map=np.zeros((1, 1)),
            magnitude_mean=0.0,
            magnitude_std=0.0,
            changed_pct=0.0,
            message=(
                "Change detection unavailable: rasters are not spatially aligned. "
                f"Shapes {a.shape} vs {b.shape}."
            ),
            valid=False,
        )

    if meta_t1 is not None and meta_t2 is not None:
        def _get(m, attr):
            if isinstance(m, dict):
                return m.get(attr)
            return getattr(m, attr, None)

        crs1, crs2 = _get(meta_t1, "crs"), _get(meta_t2, "crs")
        tr1, tr2 = _get(meta_t1, "transform"), _get(meta_t2, "transform")
        if crs1 is not None and crs2 is not None and crs1 != crs2:
            return ChangeResult(
                change_map=np.zeros((1, 1)),
                magnitude_mean=0.0,
                magnitude_std=0.0,
                changed_pct=0.0,
                message="Change detection unavailable: rasters are not spatially aligned (CRS mismatch).",
                valid=False,
            )
        if tr1 is not None and tr2 is not None:
            try:
                t1 = tuple(tr1)[:6]
                t2 = tuple(tr2)[:6]
                if any(abs(x - y) > 1e-6 for x, y in zip(t1, t2)):
                    return ChangeResult(
                        change_map=np.zeros((1, 1)),
                        magnitude_mean=0.0,
                        magnitude_std=0.0,
                        changed_pct=0.0,
                        message="Change detection unavailable: rasters are not spatially aligned (transform mismatch).",
                        valid=False,
                    )
            except Exception:
                pass

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

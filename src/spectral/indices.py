"""
Spectral Indices Engine.

Implements NDVI, NDWI and NDBI with robust handling of
division-by-zero, NaN and Inf.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Element-wise division that produces NaN on zero denominator."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(num, den)
        result[~np.isfinite(result)] = np.nan
    return result.astype(np.float32)


def _clean(arr: np.ndarray) -> np.ndarray:
    """Replace Inf with NaN and ensure float32."""
    out = arr.astype(np.float32)
    out[~np.isfinite(out)] = np.nan
    return out


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Vegetation Index.

    NDVI = (NIR - RED) / (NIR + RED)
    Theoretical range: [-1, 1]
    """
    nir = _clean(nir)
    red = _clean(red)
    return _safe_divide(nir - red, nir + red)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Water Index (McFeeters).

    NDWI = (GREEN - NIR) / (GREEN + NIR)
    Theoretical range: [-1, 1]
    """
    green = _clean(green)
    nir = _clean(nir)
    return _safe_divide(green - nir, green + nir)


def ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Built-up Index.

    NDBI = (SWIR - NIR) / (SWIR + NIR)
    Theoretical range: [-1, 1]
    """
    swir = _clean(swir)
    nir = _clean(nir)
    return _safe_divide(swir - nir, swir + nir)


def compute_all(
    bands: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Compute NDVI, NDWI, NDBI from a band dictionary.

    Expected keys: 'nir', 'red', 'green', 'swir' (or B08, B04, B03, B11).
    Missing bands are skipped; no silent invention of values.
    """
    key_map = {
        "nir": ["nir", "B08", "b08"],
        "red": ["red", "B04", "b04"],
        "green": ["green", "B03", "b03"],
        "swir": ["swir", "swir16", "B11", "b11"],
    }

    resolved: Dict[str, np.ndarray] = {}
    for canonical, aliases in key_map.items():
        for a in aliases:
            if a in bands:
                resolved[canonical] = bands[a]
                break

    results: Dict[str, np.ndarray] = {}
    if "nir" in resolved and "red" in resolved:
        results["ndvi"] = ndvi(resolved["nir"], resolved["red"])
    if "green" in resolved and "nir" in resolved:
        results["ndwi"] = ndwi(resolved["green"], resolved["nir"])
    if "swir" in resolved and "nir" in resolved:
        results["ndbi"] = ndbi(resolved["swir"], resolved["nir"])
    return results


def summarize_index(arr: np.ndarray) -> Dict[str, float]:
    """Compute robust statistics (ignoring NaN)."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "valid_pct": 0.0}
    return {
        "mean": float(np.nanmean(valid)),
        "std": float(np.nanstd(valid)),
        "min": float(np.nanmin(valid)),
        "max": float(np.nanmax(valid)),
        "valid_pct": float(100.0 * valid.size / arr.size),
    }

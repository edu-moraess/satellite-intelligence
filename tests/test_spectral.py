"""Unit tests for spectral indices."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.spectral.indices import ndvi, ndwi, ndbi, compute_all, summarize_index


def test_ndvi_range():
    nir = np.array([[0.8, 0.2], [0.5, 0.0]], dtype=np.float32)
    red = np.array([[0.2, 0.3], [0.5, 0.1]], dtype=np.float32)
    out = ndvi(nir, red)
    assert out.shape == nir.shape
    assert np.all((out[~np.isnan(out)] >= -1.0) & (out[~np.isnan(out)] <= 1.0))


def test_ndvi_zero_denom():
    nir = np.array([0.0], dtype=np.float32)
    red = np.array([0.0], dtype=np.float32)
    out = ndvi(nir, red)
    assert np.isnan(out[0])


def test_ndwi_ndbi():
    g = np.ones((4, 4), dtype=np.float32) * 0.4
    n = np.ones((4, 4), dtype=np.float32) * 0.2
    s = np.ones((4, 4), dtype=np.float32) * 0.5
    assert ndwi(g, n).shape == (4, 4)
    assert ndbi(s, n).shape == (4, 4)


def test_compute_all():
    bands = {
        "B08": np.random.rand(8, 8).astype(np.float32),
        "B04": np.random.rand(8, 8).astype(np.float32),
        "B03": np.random.rand(8, 8).astype(np.float32),
        "B11": np.random.rand(8, 8).astype(np.float32),
    }
    res = compute_all(bands)
    assert "ndvi" in res
    assert "ndwi" in res
    assert "ndbi" in res


def test_summarize():
    arr = np.array([0.1, 0.2, np.nan, 0.3])
    s = summarize_index(arr)
    assert s["valid_pct"] == 75.0
    assert abs(s["mean"] - 0.2) < 1e-5


def test_real_like_values():
    """Indices stay in [-1,1] for typical reflectance-scale inputs."""
    import numpy as np
    from src.spectral.indices import ndvi, ndwi, ndbi
    nir = np.array([[3000.0, 4000.0]], dtype=np.float32)
    red = np.array([[1500.0, 500.0]], dtype=np.float32)
    green = np.array([[1200.0, 800.0]], dtype=np.float32)
    swir = np.array([[2500.0, 3500.0]], dtype=np.float32)
    for fn, a, b in [(ndvi, nir, red), (ndwi, green, nir), (ndbi, swir, nir)]:
        out = fn(a, b)
        finite = out[np.isfinite(out)]
        assert finite.min() >= -1.0 - 1e-5
        assert finite.max() <= 1.0 + 1e-5

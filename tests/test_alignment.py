"""Tests for spatial band alignment and RGB composition."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from rasterio.transform import from_bounds
from rasterio.crs import CRS

from src.satellite.raster import (
    RasterData,
    SpatialGrid,
    align_bands,
    align_raster_to_grid,
    build_rgb,
    same_spatial_grid,
)


def _make_raster(h, w, west, south, east, north, crs="EPSG:32723", fill=100.0):
    transform = from_bounds(west, south, east, north, w, h)
    data = np.full((h, w), fill, dtype=np.float32)
    return RasterData(
        data=data,
        transform=transform,
        crs=CRS.from_string(crs),
        nodata=0.0,
        bounds=(west, south, east, north),
        band_names=["B1"],
    )


def test_align_different_resolutions():
    """10 m and 20 m bands align to same grid."""
    fine = _make_raster(4, 4, 300000, 7400000, 300040, 7400040, fill=500.0)
    coarse = _make_raster(2, 2, 300000, 7400000, 300040, 7400040, fill=800.0)
    rasters = {"B08": fine, "B11": coarse}
    aligned = align_bands(rasters, reference_key="B08")
    assert aligned["B08"].band_array().shape == aligned["B11"].band_array().shape
    assert same_spatial_grid(aligned["B08"], aligned["B11"])


def test_align_b02_b03_b04_b08_same_grid():
    bands = {
        k: _make_raster(8, 8, 0, 0, 80, 80, fill=float(i + 1) * 100)
        for i, k in enumerate(["B02", "B03", "B04", "B08"])
    }
    aligned = align_bands(bands, reference_key="B08")
    shapes = {k: v.band_array().shape for k, v in aligned.items()}
    assert len(set(shapes.values())) == 1


def test_b11_resampled_to_reference():
    ref = _make_raster(10, 10, 0, 0, 100, 100, fill=1.0)
    b11 = _make_raster(5, 5, 0, 0, 100, 100, fill=2.0)
    aligned = align_bands({"B08": ref, "B11": b11}, reference_key="B08")
    assert aligned["B11"].band_array().shape == (10, 10)
    assert same_spatial_grid(aligned["B08"], aligned["B11"])


def test_build_rgb_shape_and_range():
    h, w = 16, 16
    r = np.random.rand(h, w).astype(np.float32) * 3000
    g = np.random.rand(h, w).astype(np.float32) * 3000
    b = np.random.rand(h, w).astype(np.float32) * 3000
    rgb = build_rgb(r, g, b)
    assert rgb.shape == (h, w, 3)
    assert rgb.dtype == np.float32
    assert np.isfinite(rgb).all()
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_build_rgb_rejects_mismatch():
    r = np.ones((10, 10), dtype=np.float32)
    g = np.ones((8, 8), dtype=np.float32)
    b = np.ones((10, 10), dtype=np.float32)
    try:
        build_rgb(r, g, b)
        assert False, "Should have raised"
    except ValueError as e:
        assert "same grid" in str(e).lower() or "shape" in str(e).lower()


def test_ndvi_after_alignment():
    from src.spectral.indices import ndvi

    nir = _make_raster(8, 8, 0, 0, 80, 80, fill=3000.0)
    red = _make_raster(4, 4, 0, 0, 80, 80, fill=1500.0)
    aligned = align_bands({"B08": nir, "B04": red}, reference_key="B08")
    out = ndvi(aligned["B08"].band_array(), aligned["B04"].band_array())
    assert out.shape == (8, 8)
    finite = out[np.isfinite(out)]
    assert finite.min() >= -1.01 and finite.max() <= 1.01


def test_ndwi_ndbi_after_alignment():
    from src.spectral.indices import ndwi, ndbi

    nir = _make_raster(8, 8, 0, 0, 80, 80, fill=3000.0)
    green = _make_raster(8, 8, 0, 0, 80, 80, fill=1200.0)
    swir = _make_raster(4, 4, 0, 0, 80, 80, fill=2500.0)
    aligned = align_bands({"B08": nir, "B03": green, "B11": swir}, reference_key="B08")
    w = ndwi(aligned["B03"].band_array(), aligned["B08"].band_array())
    bi = ndbi(aligned["B11"].band_array(), aligned["B08"].band_array())
    assert w.shape == bi.shape == (8, 8)

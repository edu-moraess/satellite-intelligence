"""
Raster Engine for Satellite Intelligence.

Handles loading, validation, CRS, clipping, spatial alignment,
and RGB composition of satellite rasters. Never returns silent NaN/Inf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_geom
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry


@dataclass
class SpatialGrid:
    """Explicit spatial reference grid for band alignment."""

    crs: Any
    transform: Affine
    height: int
    width: int

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return rasterio.transform.array_bounds(self.height, self.width, self.transform)

    @property
    def resolution(self) -> Tuple[float, float]:
        return (abs(self.transform.a), abs(self.transform.e))

    def matches(self, other: "SpatialGrid", tol: float = 1e-6) -> bool:
        if self.height != other.height or self.width != other.width:
            return False
        if self.crs != other.crs:
            return False
        t1, t2 = self.transform, other.transform
        for a, b in zip(t1[:6], t2[:6]):
            if abs(a - b) > tol:
                return False
        return True


@dataclass
class RasterData:
    """Validated raster payload with spatial metadata."""

    data: np.ndarray  # (bands, height, width) or (height, width)
    transform: Any
    crs: Any
    nodata: Optional[float]
    bounds: Tuple[float, float, float, float]
    band_names: List[str]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def grid(self) -> SpatialGrid:
        if self.data.ndim == 3:
            h, w = self.data.shape[-2], self.data.shape[-1]
        else:
            h, w = self.data.shape
        return SpatialGrid(crs=self.crs, transform=self.transform, height=h, width=w)

    def is_valid(self) -> bool:
        if self.data is None or self.data.size == 0:
            return False
        if np.all(np.isnan(self.data)):
            return False
        return True

    def band_array(self) -> np.ndarray:
        """Return 2D float32 array for single-band rasters."""
        arr = self.data
        if arr.ndim == 3:
            arr = arr[0]
        return arr.astype(np.float32)


def _validate_array(arr: np.ndarray, name: str = "raster") -> np.ndarray:
    if arr is None or arr.size == 0:
        raise ValueError(f"{name}: empty array")
    if np.all(np.isnan(arr)):
        raise ValueError(f"{name}: all values are NaN")
    if np.all(np.isinf(arr)):
        raise ValueError(f"{name}: all values are Inf")
    return arr


def _to_geojson_list(aoi: Any) -> List[dict]:
    if isinstance(aoi, dict) and "type" in aoi:
        return [aoi]
    if isinstance(aoi, BaseGeometry):
        return [mapping(aoi)]
    if isinstance(aoi, (list, tuple)) and len(aoi) == 4:
        return [mapping(box(*aoi))]
    raise ValueError(
        "Unsupported AOI type; expected GeoJSON, Shapely geometry or [minx,miny,maxx,maxy]"
    )


def _downsample_with_reproject(
    data: np.ndarray,
    src_transform: Affine,
    src_crs: Any,
    out_h: int,
    out_w: int,
    resampling: Resampling = Resampling.bilinear,
) -> Tuple[np.ndarray, Affine]:
    """Geospatial downsampling via reproject (preserves transform)."""
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    n_bands, h, w = data.shape
    dst_transform = Affine(
        src_transform.a * (w / out_w),
        src_transform.b,
        src_transform.c,
        src_transform.d,
        src_transform.e * (h / out_h),
        src_transform.f,
    )
    dest = np.full((n_bands, out_h, out_w), np.nan, dtype=np.float32)
    for i in range(n_bands):
        reproject(
            source=data[i],
            destination=dest[i],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=src_crs,
            resampling=resampling,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    return dest, dst_transform


def load_raster(
    href: str,
    aoi: Optional[Any] = None,
    max_size: int = 1024,
) -> RasterData:
    """
    Load a remote or local raster, optionally clipped to AOI.

    AOI is assumed EPSG:4326 and is reprojected to the raster CRS
    before masking. Downsampling uses bilinear reproject to preserve
    geospatial transform.
    """
    with rasterio.open(href) as src:
        if src.crs is None:
            raise ValueError("Raster has no CRS")

        nodata = src.nodata

        if aoi is not None:
            geoms_wgs84 = _to_geojson_list(aoi)
            src_crs_wgs = CRS.from_epsg(4326)
            geoms = [
                transform_geom(src_crs_wgs, src.crs, g, precision=6)
                for g in geoms_wgs84
            ]
            try:
                out_image, out_transform = rio_mask(
                    src, geoms, crop=True, filled=True, all_touched=True
                )
            except ValueError as e:
                raise ValueError(
                    f"AOI does not intersect raster extent (CRS={src.crs}): {e}"
                ) from e
            data = out_image.astype(np.float32)
            transform = out_transform
        else:
            data = src.read().astype(np.float32)
            transform = src.transform

        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)

        h, w = data.shape[-2], data.shape[-1]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            out_h, out_w = max(1, int(h * scale)), max(1, int(w * scale))
            data, transform = _downsample_with_reproject(
                data, transform, src.crs, out_h, out_w, Resampling.bilinear
            )

        bounds = rasterio.transform.array_bounds(
            data.shape[-2], data.shape[-1], transform
        )
        _validate_array(data)

        band_names = (
            [f"B{i+1}" for i in range(data.shape[0])] if data.ndim == 3 else ["B1"]
        )
        return RasterData(
            data=data,
            transform=transform,
            crs=src.crs,
            nodata=nodata,
            bounds=tuple(bounds),
            band_names=band_names,
        )


def align_raster_to_grid(
    raster: RasterData,
    grid: SpatialGrid,
    resampling: Resampling = Resampling.bilinear,
) -> RasterData:
    """
    Reproject a RasterData onto an explicit SpatialGrid.

    Uses rasterio.warp.reproject — not numpy resize.
    """
    src = raster.band_array()
    dest = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    reproject(
        source=src,
        destination=dest,
        src_transform=raster.transform,
        src_crs=raster.crs,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return RasterData(
        data=dest,
        transform=grid.transform,
        crs=grid.crs,
        nodata=np.nan,
        bounds=grid.bounds,
        band_names=raster.band_names[:1] if raster.band_names else ["B1"],
    )


def align_bands(
    rasters: Dict[str, RasterData],
    reference_key: str = "B08",
) -> Dict[str, RasterData]:
    """
    Align all band rasters to the spatial grid of the reference band.

    Default reference is B08 (10 m NIR) for spectral indices.
    Falls back to the first available key if reference is missing.
    """
    if not rasters:
        raise ValueError("No rasters to align")

    if reference_key not in rasters:
        for pref in ("B08", "B04", "B03", "B02"):
            if pref in rasters:
                reference_key = pref
                break
        else:
            reference_key = next(iter(rasters))

    ref = rasters[reference_key]
    grid = ref.grid
    aligned: Dict[str, RasterData] = {}

    for key, rast in rasters.items():
        if rast.grid.matches(grid):
            aligned[key] = rast
        else:
            aligned[key] = align_raster_to_grid(rast, grid, Resampling.bilinear)

    shapes = {k: a.band_array().shape for k, a in aligned.items()}
    unique = set(shapes.values())
    if len(unique) > 1:
        raise ValueError(f"Alignment failed — residual shape mismatch: {shapes}")

    return aligned


def same_spatial_grid(a: RasterData, b: RasterData) -> bool:
    """True if two rasters share CRS, transform, height and width."""
    return a.grid.matches(b.grid)


def stack_bands(band_arrays: List[np.ndarray]) -> np.ndarray:
    """Stack single-band arrays into (C, H, W). Validates shapes."""
    if not band_arrays:
        raise ValueError("No bands provided")
    shapes = {a.shape for a in band_arrays}
    if len(shapes) > 1:
        raise ValueError(f"Band shape mismatch: {shapes}")
    stack = np.stack([_validate_array(a) for a in band_arrays], axis=0)
    return stack.astype(np.float32)


def build_rgb(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """
    Build an RGB image (H, W, 3) float32 in [0, 1] from three co-registered bands.

    Applies percentile stretch per channel. Rejects shape mismatch.
    """
    r = np.asarray(red, dtype=np.float32)
    g = np.asarray(green, dtype=np.float32)
    b = np.asarray(blue, dtype=np.float32)

    if r.ndim == 3:
        r = r[0]
    if g.ndim == 3:
        g = g[0]
    if b.ndim == 3:
        b = b[0]

    if r.shape != g.shape or r.shape != b.shape:
        raise ValueError(
            f"RGB bands must share the same grid: R{r.shape} G{g.shape} B{b.shape}"
        )

    def _stretch(ch: np.ndarray) -> np.ndarray:
        out = ch.copy()
        valid = out[np.isfinite(out)]
        if valid.size == 0:
            return np.zeros_like(out)
        lo, hi = np.percentile(valid, (p_low, p_high))
        if hi > lo:
            out = np.clip((out - lo) / (hi - lo), 0.0, 1.0)
        else:
            out = np.zeros_like(out)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.astype(np.float32)

    rgb = np.stack([_stretch(r), _stretch(g), _stretch(b)], axis=-1)
    if not np.isfinite(rgb).all():
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    return rgb

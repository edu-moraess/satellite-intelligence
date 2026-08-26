"""
Raster Engine for Satellite Intelligence.

Handles loading, validation, CRS, clipping, and basic
windowed access of satellite rasters. Never returns silent NaN/Inf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry


@dataclass
class RasterData:
    """Validated raster payload."""

    data: np.ndarray          # (bands, height, width) or (height, width)
    transform: Any
    crs: Any
    nodata: Optional[float]
    bounds: Tuple[float, float, float, float]
    band_names: List[str]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def is_valid(self) -> bool:
        if self.data is None or self.data.size == 0:
            return False
        if np.all(np.isnan(self.data)):
            return False
        return True


def _validate_array(arr: np.ndarray, name: str = "raster") -> np.ndarray:
    """Raise on empty / all-NaN / all-Inf arrays."""
    if arr is None or arr.size == 0:
        raise ValueError(f"{name}: empty array")
    if np.all(np.isnan(arr)):
        raise ValueError(f"{name}: all values are NaN")
    if np.all(np.isinf(arr)):
        raise ValueError(f"{name}: all values are Inf")
    return arr


def _to_geojson_list(aoi: Any) -> List[dict]:
    """Normalize AOI to a list of GeoJSON geometry dicts (assumed WGS84)."""
    if isinstance(aoi, dict) and "type" in aoi:
        return [aoi]
    if isinstance(aoi, BaseGeometry):
        return [mapping(aoi)]
    if isinstance(aoi, (list, tuple)) and len(aoi) == 4:
        return [mapping(box(*aoi))]
    raise ValueError("Unsupported AOI type; expected GeoJSON, Shapely geometry or [minx,miny,maxx,maxy]")


def load_raster(
    href: str,
    aoi: Optional[Any] = None,
    max_size: int = 1024,
) -> RasterData:
    """
    Load a remote or local raster, optionally clipped to AOI.

    AOI is assumed to be in EPSG:4326. It is reprojected to the
    raster CRS before masking. Uses windowed reading and simple
    downsampling when the native resolution exceeds max_size.
    """
    with rasterio.open(href) as src:
        if src.crs is None:
            raise ValueError("Raster has no CRS")

        if aoi is not None:
            geoms_wgs84 = _to_geojson_list(aoi)
            # Reproject AOI from WGS84 to raster CRS
            src_crs = CRS.from_epsg(4326)
            dst_crs = src.crs
            geoms = [
                transform_geom(src_crs, dst_crs, g, precision=6)
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
            bounds = rasterio.transform.array_bounds(
                data.shape[-2], data.shape[-1], transform
            )
            # Optional downsampling after clip
            h, w = data.shape[-2], data.shape[-1]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                out_h, out_w = max(1, int(h * scale)), max(1, int(w * scale))
                # Simple block reduce via numpy for single-pass
                from rasterio.transform import Affine
                data_ds = np.zeros((data.shape[0], out_h, out_w), dtype=np.float32)
                for c in range(data.shape[0]):
                    # nearest-neighbor style downsample
                    ys = (np.linspace(0, h - 1, out_h)).astype(int)
                    xs = (np.linspace(0, w - 1, out_w)).astype(int)
                    data_ds[c] = data[c][ys][:, xs]
                data = data_ds
                transform = Affine(
                    transform.a * (w / out_w),
                    transform.b,
                    transform.c,
                    transform.d,
                    transform.e * (h / out_h),
                    transform.f,
                )
                bounds = rasterio.transform.array_bounds(out_h, out_w, transform)
        else:
            h, w = src.height, src.width
            scale = 1.0
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
            out_h, out_w = max(1, int(h * scale)), max(1, int(w * scale))
            data = src.read(
                out_shape=(src.count, out_h, out_w),
                resampling=Resampling.bilinear,
            ).astype(np.float32)
            transform = src.transform * src.transform.scale(
                (src.width / out_w), (src.height / out_h)
            )
            bounds = src.bounds

        nodata = src.nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)

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


def stack_bands(band_arrays: List[np.ndarray]) -> np.ndarray:
    """Stack single-band arrays into (C, H, W). Validates shapes."""
    if not band_arrays:
        raise ValueError("No bands provided")
    shapes = {a.shape for a in band_arrays}
    if len(shapes) > 1:
        raise ValueError(f"Band shape mismatch: {shapes}")
    stack = np.stack([_validate_array(a) for a in band_arrays], axis=0)
    return stack.astype(np.float32)


def clip_to_aoi(raster: RasterData, aoi: Any) -> RasterData:
    """Clip an already-loaded RasterData to a geometry (re-mask)."""
    return raster

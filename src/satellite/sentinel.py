"""
Sentinel-2 specific helpers for Satellite Intelligence.

Defines collections, band nomenclature, asset keys and
metadata extraction for Sentinel-2 L2A products.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .stac import Scene

# Official Sentinel-2 L2A band mapping (Element84 / AWS naming)
SENTINEL2_BANDS: Dict[str, str] = {
    "B01": "coastal",
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B09": "nir09",
    "B11": "swir16",
    "B12": "swir22",
    "SCL": "scl",
}

# Bands commonly used for RGB visualization and spectral indices
RGB_BANDS = ["B04", "B03", "B02"]  # red, green, blue
NDVI_BANDS = ["B08", "B04"]       # nir, red
NDWI_BANDS = ["B03", "B08"]       # green, nir
NDBI_BANDS = ["B11", "B08"]       # swir, nir

# Preferred asset key aliases (Element84 uses short names)
ASSET_ALIASES: Dict[str, List[str]] = {
    "B02": ["blue", "B02"],
    "B03": ["green", "B03"],
    "B04": ["red", "B04"],
    "B08": ["nir", "B08"],
    "B11": ["swir16", "B11"],
    "B12": ["swir22", "B12"],
    "SCL": ["scl", "SCL"],
    "visual": ["visual", "overview"],
}


def get_asset_href(scene: Scene, band: str) -> Optional[str]:
    """
    Resolve the best asset href for a given Sentinel-2 band.

    Tries official band code then known aliases.
    """
    candidates = ASSET_ALIASES.get(band, [band])
    for key in candidates:
        if key in scene.assets:
            return scene.assets[key]
    # fallback: direct match
    if band in scene.assets:
        return scene.assets[band]
    return None


def extract_metadata(scene: Scene) -> Dict:
    """Return a clean metadata dictionary for UI / report."""
    return {
        "scene_id": scene.id,
        "collection": scene.collection,
        "acquisition_date": scene.datetime,
        "cloud_cover_pct": scene.cloud_cover,
        "bbox": scene.bbox,
        "available_assets": list(scene.assets.keys()),
    }


def required_bands_for_indices() -> List[str]:
    """Bands needed for NDVI / NDWI / NDBI."""
    return list({*NDVI_BANDS, *NDWI_BANDS, *NDBI_BANDS})

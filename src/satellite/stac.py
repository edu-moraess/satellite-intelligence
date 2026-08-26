"""
STAC Catalog Client for Satellite Intelligence.

Responsible for connecting to a STAC API, searching scenes,
filtering by cloud cover / datetime / AOI, and returning a
consistent scene representation. No direct raster loading here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pystac_client import Client
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry


DEFAULT_STAC_URL = "https://earth-search.aws.element84.com/v1"
DEFAULT_COLLECTION = "sentinel-2-l2a"


class SearchStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NO_RESULTS = "NO_RESULTS"
    API_ERROR = "API_ERROR"
    INVALID_AOI = "INVALID_AOI"
    INVALID_DATE_RANGE = "INVALID_DATE_RANGE"


@dataclass
class Scene:
    """Normalized scene metadata returned by STAC search."""

    id: str
    collection: str
    datetime: Optional[str]
    cloud_cover: Optional[float]
    geometry: Optional[Dict[str, Any]]
    bbox: Optional[List[float]]
    assets: Dict[str, str] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "collection": self.collection,
            "datetime": self.datetime,
            "cloud_cover": self.cloud_cover,
            "bbox": self.bbox,
            "assets": self.assets,
            "properties": self.properties,
        }


@dataclass
class SearchResult:
    """Typed STAC search outcome — distinguishes empty vs error."""

    status: SearchStatus
    scenes: List[Scene] = field(default_factory=list)
    message: str = ""
    error_detail: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == SearchStatus.SUCCESS

    def __iter__(self):
        return iter(self.scenes)

    def __len__(self) -> int:
        return len(self.scenes)

    def __bool__(self) -> bool:
        return len(self.scenes) > 0


def _aoi_to_geometry(aoi: Any) -> Dict[str, Any]:
    if isinstance(aoi, dict) and "type" in aoi:
        return aoi
    if isinstance(aoi, BaseGeometry):
        return mapping(aoi)
    if isinstance(aoi, (list, tuple)) and len(aoi) == 4:
        return mapping(box(*aoi))
    raise ValueError("AOI must be GeoJSON dict, Shapely geometry or [minx,miny,maxx,maxy]")


def search_scenes(
    aoi: Any,
    datetime_range: Optional[Tuple[str, str]] = None,
    max_cloud_cover: float = 30.0,
    limit: int = 20,
    collection: str = DEFAULT_COLLECTION,
    stac_url: str = DEFAULT_STAC_URL,
) -> SearchResult:
    """
    Search STAC for Sentinel-2 scenes intersecting the AOI.

    Returns SearchResult with explicit status:
    SUCCESS | NO_RESULTS | API_ERROR | INVALID_AOI | INVALID_DATE_RANGE
    """
    try:
        geom = _aoi_to_geometry(aoi)
    except Exception as e:
        return SearchResult(
            status=SearchStatus.INVALID_AOI,
            message="Invalid area of interest.",
            error_detail=str(e),
        )

    try:
        if datetime_range is None:
            end = datetime.utcnow()
            start = end - timedelta(days=180)
            datetime_str = f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        else:
            datetime_str = f"{datetime_range[0]}/{datetime_range[1]}"
    except Exception as e:
        return SearchResult(
            status=SearchStatus.INVALID_DATE_RANGE,
            message="Invalid date range.",
            error_detail=str(e),
        )

    try:
        client = Client.open(stac_url)
        search = client.search(
            collections=[collection],
            intersects=geom,
            datetime=datetime_str,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            max_items=limit,
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        )

        scenes: List[Scene] = []
        for item in search.items():
            props = dict(item.properties) if item.properties else {}
            cloud = props.get("eo:cloud_cover")
            assets = {}
            for key, asset in item.assets.items():
                if asset.href:
                    assets[key] = asset.href
            scenes.append(
                Scene(
                    id=item.id,
                    collection=item.collection_id or collection,
                    datetime=props.get("datetime"),
                    cloud_cover=float(cloud) if cloud is not None else None,
                    geometry=item.geometry,
                    bbox=list(item.bbox) if item.bbox else None,
                    assets=assets,
                    properties=props,
                )
            )

        if not scenes:
            return SearchResult(
                status=SearchStatus.NO_RESULTS,
                scenes=[],
                message="No suitable Sentinel-2 scenes found for the given AOI and filters.",
            )
        return SearchResult(
            status=SearchStatus.SUCCESS,
            scenes=scenes,
            message=f"Found {len(scenes)} scene(s).",
        )
    except Exception as e:
        return SearchResult(
            status=SearchStatus.API_ERROR,
            scenes=[],
            message="STAC API request failed.",
            error_detail=str(e),
        )


def select_best_scene(scenes) -> Optional[Scene]:
    """Select the scene with lowest cloud cover (already sorted)."""
    if isinstance(scenes, SearchResult):
        scenes = scenes.scenes
    if not scenes:
        return None
    return scenes[0]

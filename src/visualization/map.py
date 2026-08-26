"""
Map visualization helpers for Satellite Intelligence V2.

Map is independent of STAC / Sentinel-2 / Deep Learning.
Always renders with stable key and defined height.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import folium
from folium.plugins import Draw
from shapely.geometry import mapping

# Default view: São Paulo
DEFAULT_CENTER: Tuple[float, float] = (-23.5505, -46.6333)
DEFAULT_ZOOM: int = 11
MAP_HEIGHT: int = 420


def create_base_map(
    center: Tuple[float, float] = DEFAULT_CENTER,
    zoom: int = DEFAULT_ZOOM,
) -> folium.Map:
    """Create a clean base map with OSM tiles (always visible)."""
    lat, lon = center
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        lat, lon = DEFAULT_CENTER
    m = folium.Map(
        location=[lat, lon],
        zoom_start=max(1, min(int(zoom), 18)),
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )
    folium.TileLayer("CartoDB positron", name="Light", control=True).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def add_aoi(m: folium.Map, geometry: Any, name: str = "AOI") -> folium.Map:
    """Overlay AOI geometry (EPSG:4326 GeoJSON or Shapely)."""
    if geometry is None:
        return m
    if isinstance(geometry, dict):
        geojson = geometry
    else:
        try:
            geojson = mapping(geometry)
        except Exception:
            return m
    folium.GeoJson(
        geojson,
        name=name,
        style_function=lambda _x: {
            "fillColor": "#2563eb",
            "color": "#1d4ed8",
            "weight": 2,
            "fillOpacity": 0.2,
        },
    ).add_to(m)
    return m


def add_draw_control(m: folium.Map) -> folium.Map:
    """Rectangle + Polygon draw tools."""
    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": True,
            "rectangle": True,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)
    return m


def add_marker(m: folium.Map, lat: float, lon: float, popup: str = "") -> folium.Map:
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return m
    folium.Marker(
        location=[lat, lon],
        popup=popup or f"{lat:.4f}, {lon:.4f}",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)
    return m

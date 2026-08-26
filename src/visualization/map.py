"""
Map visualization helpers for Satellite Intelligence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import folium
from folium.plugins import Draw
from shapely.geometry import mapping, shape


def create_base_map(
    center: Tuple[float, float] = (0.0, 20.0),
    zoom: int = 2,
    height: str = "500px",
) -> folium.Map:
    """Create a clean dark-themed base map."""
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
    )
    return m


def add_aoi(m: folium.Map, geometry: Any, name: str = "AOI") -> folium.Map:
    """Add AOI geometry to the map."""
    if geometry is None:
        return m
    if isinstance(geometry, dict):
        geojson = geometry
    else:
        geojson = mapping(geometry)
    folium.GeoJson(
        geojson,
        name=name,
        style_function=lambda x: {
            "fillColor": "#1a73e8",
            "color": "#1a73e8",
            "weight": 2,
            "fillOpacity": 0.15,
        },
    ).add_to(m)
    return m


def add_draw_control(m: folium.Map) -> folium.Map:
    """Enable rectangle / polygon drawing."""
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
        edit_options={"edit": True},
    ).add_to(m)
    return m


def add_marker(m: folium.Map, lat: float, lon: float, popup: str = "") -> folium.Map:
    folium.Marker(
        location=[lat, lon],
        popup=popup,
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)
    return m

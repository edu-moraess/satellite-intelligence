"""
EXPLORE UI – Location selection.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import streamlit as st
from shapely.geometry import box, mapping, Point

from src.catalog.locations import Location, search_locations


def _init_explore_state():
    defaults = {
        "selected_location": None,
        "aoi_geometry": None,
        "aoi_center": None,
        "aoi_zoom": 10,
        "manual_lat": 0.0,
        "manual_lon": 0.0,
        "aoi_size_km": 5.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _set_location(loc: Location, size_km: float = 5.0):
    """Configure session state from a catalog location."""
    st.session_state.selected_location = loc
    half = size_km / 111.0  # rough degrees
    geom = box(loc.longitude - half, loc.latitude - half, loc.longitude + half, loc.latitude + half)
    st.session_state.aoi_geometry = mapping(geom)
    st.session_state.aoi_center = (loc.latitude, loc.longitude)
    st.session_state.aoi_zoom = 12


def _set_manual(lat: float, lon: float, size_km: float):
    half = size_km / 111.0
    geom = box(lon - half, lat - half, lon + half, lat + half)
    st.session_state.aoi_geometry = mapping(geom)
    st.session_state.aoi_center = (lat, lon)
    st.session_state.aoi_zoom = 12
    st.session_state.selected_location = Location(
        id="manual",
        country="Custom",
        country_code="XX",
        region="Custom",
        location=f"{lat:.4f}, {lon:.4f}",
        category="CUSTOM",
        latitude=lat,
        longitude=lon,
    )


def render_explore() -> bool:
    """
    Render the EXPLORE section.
    Returns True when a valid AOI is ready for analysis.
    """
    _init_explore_state()

    st.markdown("## Where do you want to investigate?")
    st.caption("Select a location from the global catalog, enter coordinates, or define an AOI.")

    tab_search, tab_manual, tab_map = st.tabs(["Global Catalog", "Manual Coordinates", "Draw on Map"])

    with tab_search:
        query = st.text_input("Search location", placeholder="Tokyo, São Paulo, Mount Fuji, Amazon…", key="loc_search")
        if query:
            results = search_locations(query, limit=12)
            if not results:
                st.info("No matching locations found.")
            else:
                for loc in results:
                    cols = st.columns([3, 2, 2, 1])
                    cols[0].markdown(f"**{loc.location}**")
                    cols[1].markdown(f"{loc.country}")
                    cols[2].markdown(f"`{loc.category}`  \n{loc.coordinates_str}")
                    if cols[3].button("SELECT", key=f"sel_{loc.id}"):
                        _set_location(loc)
                        st.rerun()

    with tab_manual:
        c1, c2, c3 = st.columns(3)
        lat = c1.number_input("Latitude", min_value=-90.0, max_value=90.0, value=st.session_state.manual_lat, format="%.6f")
        lon = c2.number_input("Longitude", min_value=-180.0, max_value=180.0, value=st.session_state.manual_lon, format="%.6f")
        size = c3.number_input("AOI size (km)", min_value=0.5, max_value=50.0, value=st.session_state.aoi_size_km, step=0.5)
        if st.button("Set Coordinates", type="primary"):
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180) or size <= 0:
                st.error("Invalid coordinates or AOI size.")
            else:
                st.session_state.manual_lat = lat
                st.session_state.manual_lon = lon
                st.session_state.aoi_size_km = size
                _set_manual(lat, lon, size)
                st.success(f"AOI set around {lat:.4f}, {lon:.4f}")
                st.rerun()

    with tab_map:
        st.info("Draw a rectangle or polygon on the map below. The drawn geometry becomes the AOI.")
        # Placeholder for map – actual interactive draw needs streamlit-folium
        try:
            from streamlit_folium import st_folium
            from src.visualization.map import create_base_map, add_draw_control

            center = st.session_state.aoi_center or (0.0, 20.0)
            zoom = st.session_state.aoi_zoom or 2
            m = create_base_map(center=center, zoom=zoom)
            m = add_draw_control(m)
            out = st_folium(m, width=None, height=420, key="draw_map")
            if out and out.get("all_drawings"):
                drawings = out["all_drawings"]
                if drawings:
                    geom = drawings[-1].get("geometry")
                    if geom:
                        st.session_state.aoi_geometry = geom
                        # approximate center
                        from shapely.geometry import shape as sh
                        s = sh(geom)
                        cen = s.centroid
                        st.session_state.aoi_center = (cen.y, cen.x)
                        st.session_state.selected_location = Location(
                            id="drawn",
                            country="Custom",
                            country_code="XX",
                            region="Custom",
                            location="Drawn AOI",
                            category="CUSTOM",
                            latitude=cen.y,
                            longitude=cen.x,
                        )
                        st.success("AOI captured from drawing.")
        except ImportError:
            st.warning("streamlit-folium not available. Use Catalog or Manual Coordinates.")

    # Summary of current selection
    loc = st.session_state.selected_location
    if loc is not None and st.session_state.aoi_geometry is not None:
        st.divider()
        st.markdown(f"### LOCATION  \n**{loc.display_name}**")
        st.markdown(f"Coordinates  \n`{loc.latitude:.4f}`  `{loc.longitude:.4f}`")
        return True
    return False

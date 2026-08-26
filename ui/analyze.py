"""
ANALYZE UI – Satellite data acquisition and spectral analysis.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import streamlit as st

from src.satellite.stac import search_scenes, select_best_scene, Scene, SearchStatus
from src.satellite.sentinel import get_asset_href, extract_metadata
from src.satellite.raster import load_raster, align_bands, build_rgb, RasterData
from src.spectral.indices import compute_all, summarize_index
from src.visualization.charts import index_summary_chart


def render_analyze() -> None:
    """Render ANALYZE section once an AOI is available."""
    aoi = st.session_state.get("aoi_geometry")
    if aoi is None:
        st.info("Select a location first.")
        return

    st.markdown("## Satellite Data")

    col1, col2, col3 = st.columns(3)
    max_cloud = col1.slider("Max cloud cover (%)", 0, 100, 30)
    days_back = col2.slider("Look-back window (days)", 30, 365, 180)
    limit = col3.slider("Max scenes", 5, 50, 15)

    if st.button("Search Sentinel-2", type="primary"):
        with st.spinner("Querying STAC catalog…"):
            end = datetime.utcnow()
            start = end - timedelta(days=days_back)
            result = search_scenes(
                aoi=aoi,
                datetime_range=(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
                max_cloud_cover=float(max_cloud),
                limit=limit,
            )
            st.session_state.stac_result = result
            st.session_state.scenes = result.scenes
            st.session_state.selected_scene = select_best_scene(result)

    result = st.session_state.get("stac_result")
    scenes = st.session_state.get("scenes", [])
    scene: Optional[Scene] = st.session_state.get("selected_scene")

    if result is not None:
        if result.status == SearchStatus.API_ERROR:
            st.error("STAC API request failed.")
            if result.error_detail:
                st.caption(f"Detail: {result.error_detail}")
            return
        if result.status == SearchStatus.INVALID_AOI:
            st.error("Invalid area of interest.")
            return
        if result.status == SearchStatus.NO_RESULTS:
            st.warning(result.message or "No suitable Sentinel-2 scene was found.")
            return

    if not scenes and "scenes" in st.session_state:
        st.warning("No suitable Sentinel-2 scene was found.")
        return

    if scene is None:
        st.info("Run a search to discover Sentinel-2 scenes.")
        return

    st.success(
        f"Selected scene: **{scene.id}**  |  Cloud: {scene.cloud_cover:.1f}%  |  Date: {scene.datetime}"
    )

    if st.button("Load preview & compute indices"):
        with st.spinner("Loading and aligning raster assets…"):
            try:
                rasters: dict = {}
                load_errors = []
                for b in ["B04", "B03", "B02", "B08", "B11"]:
                    href = get_asset_href(scene, b)
                    if not href:
                        load_errors.append(f"{b}: asset not found")
                        continue
                    try:
                        rasters[b] = load_raster(href, aoi=aoi, max_size=512)
                    except Exception as band_err:
                        load_errors.append(f"{b}: {band_err}")

                if not rasters:
                    st.error("Could not load required band assets.")
                    if load_errors:
                        st.caption("Details: " + " | ".join(load_errors[:3]))
                    return

                # Spatial alignment to B08 grid (10 m)
                aligned = align_bands(rasters, reference_key="B08")
                bands = {k: r.band_array() for k, r in aligned.items()}

                # RGB composition
                rgb = None
                if all(k in bands for k in ("B04", "B03", "B02")):
                    rgb = build_rgb(bands["B04"], bands["B03"], bands["B02"])

                # Spectral indices
                indices = compute_all(bands)
                summaries = {k: summarize_index(v) for k, v in indices.items()}

                st.session_state.band_data = bands
                st.session_state.raster_meta = {
                    k: {"crs": str(r.crs), "shape": r.band_array().shape, "transform": r.transform}
                    for k, r in aligned.items()
                }
                st.session_state.rgb_image = rgb
                st.session_state.spectral_summaries = summaries
                st.session_state.spectral_indices = indices
                st.session_state.analysis_ready = True
                st.success("Spectral analysis complete.")
            except Exception as e:
                st.error(f"Raster processing failed: {e}")

    if st.session_state.get("analysis_ready"):
        rgb = st.session_state.get("rgb_image")
        if rgb is not None:
            st.markdown("### Sentinel-2 RGB")
            st.image(rgb, caption="True-color composite (B04 / B03 / B02)", use_container_width=True)

        summaries = st.session_state.get("spectral_summaries", {})
        if summaries:
            st.markdown("### Key Metrics")
            cols = st.columns(len(summaries))
            for i, (name, stats) in enumerate(summaries.items()):
                with cols[i]:
                    st.metric(name.upper(), f"{stats['mean']:.3f}", help=f"std={stats['std']:.3f}")
            fig = index_summary_chart(summaries)
            st.plotly_chart(fig, use_container_width=True)

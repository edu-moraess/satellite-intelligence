"""
REPORT UI – Geospatial intelligence summary.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st


def render_report() -> None:
    st.markdown("## Geospatial Intelligence Report")

    loc = st.session_state.get("selected_location")
    scene = st.session_state.get("selected_scene")
    summaries = st.session_state.get("spectral_summaries", {})
    dl = st.session_state.get("deep_learning_result")

    if loc is None:
        st.info("No analysis has been performed yet.")
        return

    st.markdown(f"**Location:** {loc.display_name}")
    st.markdown(f"**Coordinates:** {loc.latitude:.4f}, {loc.longitude:.4f}")
    st.markdown(f"**Category:** {loc.category}")
    st.markdown(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    st.divider()

    if scene:
        st.markdown("### Satellite Scene")
        st.markdown(f"- **ID:** `{scene.id}`")
        st.markdown(f"- **Acquisition:** {scene.datetime}")
        st.markdown(f"- **Cloud cover:** {scene.cloud_cover:.1f}%")

    if summaries:
        st.markdown("### Spectral Indicators")
        for name, stats in summaries.items():
            st.markdown(f"- **{name.upper()}** mean = `{stats['mean']:.3f}` (valid {stats['valid_pct']:.0f}%)")

    if dl:
        st.markdown("### Deep Learning")
        if dl.model_available:
            st.markdown("Model: U-Net · Task: Semantic Segmentation")
            for name, pct in dl.class_percentages.items():
                st.markdown(f"- {name}: **{pct:.1f}%**")
        else:
            st.markdown(f"*{dl.message}*")

    st.divider()
    st.caption("Satellite Intelligence · Earth Observation & Deep Geospatial Intelligence")

"""
REPORT UI – Geospatial intelligence summary.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from src.deep_learning.inference.segmentation import ModelStatus


def render_report() -> None:
    st.markdown("## Geospatial Intelligence Report")

    loc = st.session_state.get("selected_location")
    scene = st.session_state.get("selected_scene")
    summaries = st.session_state.get("spectral_summaries", {})
    dl = st.session_state.get("deep_learning_result")
    has_rgb = st.session_state.get("rgb_image") is not None

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
        st.markdown(f"- **RGB composite:** {'Available' if has_rgb else 'Not loaded'}")

    if summaries:
        st.markdown("### Spectral Indicators")
        for name, stats in summaries.items():
            st.markdown(
                f"- **{name.upper()}** mean = `{stats['mean']:.3f}` "
                f"(valid {stats['valid_pct']:.0f}%)"
            )

    st.markdown("### Deep Learning")
    if dl is None:
        st.markdown("- **Status:** Unavailable")
        st.markdown("- **Reason:** U-Net checkpoint not included in V1")
        st.markdown("- **Inference:** Not executed")
    else:
        status = getattr(dl, "status", None)
        if status == ModelStatus.SUCCESS and dl.model_available:
            st.markdown("- **Status:** Success")
            st.markdown("- Model: U-Net · Task: Semantic Segmentation")
            for name, pct in dl.class_percentages.items():
                st.markdown(f"- {name}: **{pct:.1f}%**")
        elif status == ModelStatus.MODEL_LOAD_ERROR:
            st.markdown("- **Status:** Load error")
            st.markdown(f"- **Reason:** {dl.message}")
            st.markdown("- **Inference:** Not executed")
        elif status == ModelStatus.MODEL_INFERENCE_ERROR:
            st.markdown("- **Status:** Inference error")
            st.markdown(f"- **Reason:** {dl.message}")
        else:
            st.markdown("- **Status:** Unavailable")
            st.markdown("- **Reason:** U-Net checkpoint not included in V1")
            st.markdown("- **Inference:** Not executed")

    st.divider()
    st.caption("Satellite Intelligence · Earth Observation & Deep Geospatial Intelligence")

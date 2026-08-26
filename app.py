"""
SATELLITE INTELLIGENCE
Earth Observation & Deep Geospatial Intelligence

Entry point – keeps orchestration only.
All domain logic lives under src/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from ui.explore import render_explore
from ui.analyze import render_analyze
from ui.intelligence import render_intelligence
from ui.report import render_report


def _inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*=\"css\"] {
            font-family: 'Inter', system-ui, sans-serif;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }
        h1 {
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
        }
        h2 {
            font-weight: 600;
            margin-top: 1.5rem;
        }
        .stMetric {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.75rem 1rem;
        }
        div[data-testid=\"stSidebar\"] {
            background: #0f172a;
        }
        div[data-testid=\"stSidebar\"] * {
            color: #e2e8f0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Satellite Intelligence",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    # Header
    st.markdown("# SATELLITE INTELLIGENCE")
    st.markdown("**Earth Observation & Deep Geospatial Intelligence**")
    st.caption("Explore · Select · Analyze · Deep Intelligence · Report")

    # Sidebar – minimal controls only
    with st.sidebar:
        st.markdown("### Navigation")
        section = st.radio(
            "Section",
            ["EXPLORE", "ANALYZE", "DEEP INTELLIGENCE", "REPORT"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("### System")
        st.caption("Sentinel-2 L2A via STAC")
        st.caption("U-Net Semantic Segmentation")
        st.caption("NDVI · NDWI · NDBI")
        if st.session_state.get("selected_location"):
            loc = st.session_state.selected_location
            st.divider()
            st.markdown("**Active AOI**")
            st.caption(loc.display_name)
            st.caption(f"{loc.latitude:.4f}, {loc.longitude:.4f}")

    # Routing
    if section == "EXPLORE":
        ready = render_explore()
        if ready:
            st.info("AOI ready. Switch to ANALYZE to search Sentinel-2.")
    elif section == "ANALYZE":
        render_analyze()
    elif section == "DEEP INTELLIGENCE":
        render_intelligence()
    elif section == "REPORT":
        render_report()


if __name__ == "__main__":
    main()

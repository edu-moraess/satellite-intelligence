"""
DEEP INTELLIGENCE UI – U-Net semantic segmentation.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.deep_learning.inference.segmentation import run_segmentation
from src.visualization.charts import class_distribution_chart


def render_intelligence() -> None:
    st.markdown("## Deep Intelligence")
    st.caption("U-Net Semantic Segmentation · 6 land-cover classes")

    bands = st.session_state.get("band_data")
    if not bands:
        st.info("Complete spectral analysis first to obtain input imagery.")
        return

    # Build a 3-band input (R, G, B preferred)
    order = ["B04", "B03", "B02"]
    available = [b for b in order if b in bands]
    if len(available) < 3:
        # fallback any 3
        available = list(bands.keys())[:3]
    if len(available) < 1:
        st.error("No band data available for inference.")
        return

    stacked = np.stack([bands[b] for b in available], axis=0)

    if st.button("Run Segmentation", type="primary"):
        with st.spinner("Running U-Net inference…"):
            result = run_segmentation(stacked)
            st.session_state.deep_learning_result = result

    result = st.session_state.get("deep_learning_result")
    if result is None:
        return

    if not result.model_available:
        st.warning(result.message)
        st.info(
            "Place a trained checkpoint at `models/checkpoints/unet_sentinel2.pt` "
            "to enable Deep Learning. The rest of the platform continues to work."
        )
        return

    st.success(result.message)

    st.markdown("### Land Cover Summary")
    cols = st.columns(3)
    for i, (name, pct) in enumerate(result.class_percentages.items()):
        cols[i % 3].metric(name, f"{pct:.1f}%")

    fig = class_distribution_chart(result.class_percentages)
    st.plotly_chart(fig, use_container_width=True)

    # Show segmentation map as image
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        colors = ["#6b7280", "#22c55e", "#3b82f6", "#ef4444", "#d97706", "#a3e635"]
        cmap = ListedColormap(colors)
        fig2, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(result.segmentation, cmap=cmap, vmin=0, vmax=5)
        ax.set_title("Segmentation Map")
        ax.axis("off")
        st.pyplot(fig2)
        plt.close(fig2)
    except Exception:
        st.image(result.segmentation, caption="Segmentation Map", use_container_width=True)

"""
DEEP INTELLIGENCE UI – U-Net semantic segmentation.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.deep_learning.inference.segmentation import (
    run_segmentation,
    ModelStatus,
    _find_checkpoint,
)
from src.deep_learning.models.unet import build_unet, NUM_CLASSES
from src.visualization.charts import class_distribution_chart


def render_intelligence() -> None:
    st.markdown("## Deep Intelligence")
    st.caption("U-Net Semantic Segmentation · 6 land-cover classes")

    arch_ok = True
    try:
        m = build_unet(in_channels=3, num_classes=NUM_CLASSES, base_filters=16)
        _ = m
    except Exception:
        arch_ok = False

    ckpt = _find_checkpoint()
    ckpt_ok = ckpt is not None

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**U-Net Architecture**  \n{'✓' if arch_ok else '✗'}")
    c2.markdown("**Preprocessing**  \n✓")
    c3.markdown("**Inference Pipeline**  \n✓")
    c4.markdown(f"**Checkpoint**  \n{'✓' if ckpt_ok else '—'}")

    bands = st.session_state.get("band_data")
    if not bands:
        st.info("Complete spectral analysis first to obtain input imagery.")
        if not ckpt_ok:
            st.markdown("### STATUS: **UNAVAILABLE**")
            st.caption("U-Net checkpoint is not included in V1.")
            st.caption("Inference: Not executed")
        return

    order = ["B04", "B03", "B02"]
    available = [b for b in order if b in bands]
    if len(available) < 3:
        available = list(bands.keys())[:3]
    if len(available) < 1:
        st.error("No band data available for inference.")
        return

    shapes = {b: bands[b].shape for b in available}
    if len(set(shapes.values())) > 1:
        st.error(f"Bands not spatially aligned for U-Net input: {shapes}")
        return

    stacked = np.stack([bands[b] for b in available], axis=0)

    if st.button("Run Segmentation", type="primary"):
        with st.spinner("Running U-Net inference…"):
            result = run_segmentation(stacked)
            st.session_state.deep_learning_result = result

    result = st.session_state.get("deep_learning_result")
    if result is None:
        if not ckpt_ok:
            st.markdown("### STATUS: **UNAVAILABLE**")
            st.caption("U-Net checkpoint is not included in V1.")
            st.caption("Inference: Not executed")
        return

    status = getattr(result, "status", None)
    if status == ModelStatus.MODEL_UNAVAILABLE or not result.model_available:
        st.markdown("### STATUS: **UNAVAILABLE**")
        st.warning(result.message)
        st.caption("Inference: Not executed")
        return
    if status == ModelStatus.MODEL_LOAD_ERROR:
        st.markdown("### STATUS: **LOAD ERROR**")
        st.error(result.message)
        return
    if status == ModelStatus.MODEL_INFERENCE_ERROR:
        st.markdown("### STATUS: **INFERENCE ERROR**")
        st.error(result.message)
        return

    st.markdown("### STATUS: **SUCCESS**")
    st.success(result.message)

    st.markdown("### Land Cover Summary")
    cols = st.columns(3)
    for i, (name, pct) in enumerate(result.class_percentages.items()):
        cols[i % 3].metric(name, f"{pct:.1f}%")

    fig = class_distribution_chart(result.class_percentages)
    st.plotly_chart(fig, use_container_width=True)

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

"""
Chart helpers for spectral and segmentation summaries.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import plotly.graph_objects as go
import plotly.express as px


# Professional color palette
CLASS_COLORS = {
    "Background": "#6b7280",
    "Vegetation": "#22c55e",
    "Water": "#3b82f6",
    "Urban": "#ef4444",
    "Bare Soil": "#d97706",
    "Agriculture": "#a3e635",
}


def class_distribution_chart(percentages: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart of class percentages."""
    names = list(percentages.keys())
    values = list(percentages.values())
    colors = [CLASS_COLORS.get(n, "#94a3b8") for n in names]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Land Cover Distribution",
        xaxis_title="Percentage (%)",
        yaxis_title="",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    return fig


def index_summary_chart(summaries: Dict[str, Dict[str, float]]) -> go.Figure:
    """Bar chart of mean spectral index values."""
    indices = list(summaries.keys())
    means = [summaries[k].get("mean", 0.0) for k in indices]

    fig = go.Figure(
        go.Bar(
            x=indices,
            y=means,
            marker_color=["#22c55e", "#3b82f6", "#f59e0b"],
            text=[f"{v:.3f}" for v in means],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Spectral Indices (Mean)",
        yaxis_title="Index Value",
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[-1, 1]),
    )
    return fig

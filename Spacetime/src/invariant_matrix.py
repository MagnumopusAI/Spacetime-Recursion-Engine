"""Invariant Forms Matrix Visualization.

This module constructs a heatmap highlighting physical invariants across
multiple domains. The approach mirrors how geographers overlay data
layers on a map to reveal hidden relationships.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go


def get_invariant_forms() -> Tuple[List[str], List[str], list]:
    """Return the domain labels, characteristics, and data table.

    The dataset encodes how certain quadratic forms appear in different
    areas of physics. Think of it like cataloging landmarks along a
    landscape where the Preservation Constraint Equation (PCE) holds.
    """

    domains = [
        "Differential Geometry",
        "Particle Physics",
        "Quantum Mechanics",
        "Fluid/Plasma Dynamics",
    ]
    characteristics = [
        "Quadratic Form",
        "Eigenmode Type",
        "Physical Role",
        "λ=4 Compatible",
    ]

    data = [
        [
            {"value": "K = R_uv R^uv", "color": "blue"},
            {"value": "Real", "color": "green"},
            {"value": "Curvature", "color": "yellow"},
            {"value": "Yes", "color": "green"},
        ],
        [
            {"value": "F_μν F^μν", "color": "blue"},
            {"value": "Complex", "color": "red"},
            {"value": "Field Energy", "color": "yellow"},
            {"value": "Yes", "color": "green"},
        ],
        [
            {"value": "ΔE ∝ E²", "color": "blue"},
            {"value": "Real/Complex", "color": "red"},
            {"value": "Energy Shift", "color": "yellow"},
            {"value": "Partial", "color": "yellow"},
        ],
        [
            {"value": "ω(k) dispersion", "color": "red"},
            {"value": "Complex", "color": "red"},
            {"value": "Wave Prop", "color": "red"},
            {"value": "Yes", "color": "green"},
        ],
    ]
    return domains, characteristics, data


def build_color_map() -> dict:
    """Create a mapping from semantic colors to brand hex codes."""
    return {
        "green": "#1FB8CD",  # teal-like branding
        "blue": "#FFC185",
        "red": "#B4413C",
        "yellow": "#D2BA4C",
    }


def convert_to_matrices(
    domains: List[str],
    characteristics: List[str],
    data: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert data table to text and numeric matrices.

    Parameters
    ----------
    domains:
        Labels for each physics domain.
    characteristics:
        Column labels describing mathematical properties.
    data:
        Nested list with values and colors.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Text matrix and numeric z-matrix used for heatmap coloring.
    """

    text_matrix = []
    z_matrix = []
    color_map = build_color_map()

    color_scale = {"green": 3, "blue": 2, "yellow": 1, "red": 0}

    for row in data:
        text_row = []
        z_row = []
        for cell in row:
            text_row.append(cell["value"])
            z_row.append(color_scale[cell["color"]])
        text_matrix.append(text_row)
        z_matrix.append(z_row)

    return np.array(text_matrix), np.array(z_matrix)


def generate_invariant_heatmap() -> go.Figure:
    """Generate and return the invariant forms heatmap figure."""

    domains, characteristics, data = get_invariant_forms()
    text_matrix, z_matrix = convert_to_matrices(domains, characteristics, data)
    color_map = build_color_map()

    colorscale = [
        [0.0, color_map["red"]],
        [0.33, color_map["yellow"]],
        [0.66, color_map["blue"]],
        [1.0, color_map["green"]],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=characteristics,
            y=domains,
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 12, "color": "white"},
            colorscale=colorscale,
            showscale=False,
            hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Invariant Mathematical Forms Matrix",
        xaxis_title="Math Characteristics",
        yaxis_title="Physics Domains",
    )
    fig.update_xaxes(side="top")
    fig.update_yaxes(autorange="reversed")

    return fig


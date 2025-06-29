"""Plotting helpers for eigenmode selection.

This module builds a flow-chart style diagram showing how the
``lambda = 4`` eigenmode emerges from the Preservation Constraint
Equation (PCE).  The diagram mirrors a physical filter that only
allows the resonant mode to pass, akin to tuning forks or spectral
filters in optics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .preservation import compute_lambda_4_eigenmode


def build_eigenmode_filter_diagram(
    eigenvalues: Iterable[int],
    selected_mode: int = 4,
    input_heights: Optional[Iterable[float]] = None,
) -> go.Figure:
    """Return a Plotly figure illustrating eigenmode selection.

    Parameters
    ----------
    eigenvalues:
        Sequence of candidate eigenmode labels.
    selected_mode:
        The ``lambda`` value passing through the filter.
    input_heights:
        Optional relative amplitudes for the input spectrum.

    Returns
    -------
    go.Figure
        Visualization tracing input modes through a ``lambda=4`` filter.

    Notes
    -----
    This function invokes :func:`compute_lambda_4_eigenmode` to
    symbolically enforce the PCE before plotting.  The returned
    figure resembles a resonant cavity that isolates the desired
    frequency while damping others.
    """

    eigenvalues = list(eigenvalues)
    if input_heights is None:
        input_heights = [1.0 for _ in eigenvalues]
    else:
        input_heights = list(input_heights)

    # Ensure PCE invariants are respected for demonstration
    compute_lambda_4_eigenmode(sigma=0.1, tau=0.2)

    colors = ["#5D878F" for _ in eigenvalues]
    try:
        idx = eigenvalues.index(selected_mode)
        colors[idx] = "#1FB8CD"
    except ValueError:
        pass

    fig = go.Figure()

    for i, (eigenval, height, color) in enumerate(zip(eigenvalues, input_heights, colors)):
        fig.add_bar(
            x=[i - 2],
            y=[height],
            name=f"λ={eigenval}",
            marker_color=color,
            opacity=0.8 if eigenval != selected_mode else 1.0,
            text=f"λ={eigenval}",
            textposition="outside",
        )

    fig.add_shape(
        type="rect",
        x0=1.5,
        y0=0.2,
        x1=4.5,
        y1=1.8,
        line=dict(color="#13343B", width=3),
        fillcolor="#ECEBD5",
        opacity=0.3,
    )

    fig.add_bar(
        x=[6],
        y=[1.5],
        name="Output λ=4",
        marker_color="#1FB8CD",
        text="λ=4 Selected",
        textposition="outside",
    )

    blocked_positions = np.linspace(5.2, 6.8, len(eigenvalues))
    for pos, eigenval in zip(blocked_positions, eigenvalues):
        if eigenval == selected_mode:
            continue
        fig.add_bar(
            x=[pos],
            y=[0.2],
            name=f"Blocked λ={eigenval}",
            marker_color="#5D878F",
            opacity=0.3,
            text=f"λ={eigenval}",
            textposition="outside",
        )

    fig.update_layout(
        xaxis=dict(range=[-3, 8], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(range=[-1, 2.2], title="Amplitude", showgrid=True, gridcolor="lightgray"),
        plot_bgcolor="white",
    )

    fig.add_annotation(x=-1, y=2.0, text="Input Spectrum", showarrow=False)
    fig.add_annotation(x=3, y=2.0, text="λ=4 Filter", showarrow=False)
    fig.add_annotation(x=6, y=2.0, text="Output", showarrow=False)

    return fig


def save_diagram(fig: go.Figure, path: Path) -> Path:
    """Save a diagram to ``path`` using the Kaleido engine.

    Parameters
    ----------
    fig:
        Figure produced by :func:`build_eigenmode_filter_diagram`.
    path:
        Output file location, typically ending with ``.png``.

    Returns
    -------
    Path
        The path that was written.

    This helper mirrors exporting an experimental plot after verifying
    a solution satisfies the PCE.
    """

    fig.write_image(str(path))
    return path


def pattern_recognition_vs_discovery() -> go.Figure:
    """Return a chart contrasting reactive recognition with proactive discovery.

    This visualization juxtaposes two learning modes like comparing
    reflexive muscle memory to deliberate exploration. Each column lists
    qualitative traits that either interpolate within known patterns or
    extrapolate toward novel structure.

    Returns
    -------
    go.Figure
        Bar chart summarizing pattern recognition and genuine discovery.
    """

    pr_chars = [
        "Interpolates",
        "Matches templ",
        "High confid",
        "Fails novelty",
        "Overfits corr",
        "Reactive",
    ]

    gd_chars = [
        "Extrapolates",
        "Creates framew",
        "Approp uncert",
        "Handles compl",
        "Ident causal",
        "Proactive",
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Pattern Recog", "Genuine Disc"),
        horizontal_spacing=0.15,
    )

    fig.add_trace(
        go.Bar(
            y=list(range(len(pr_chars))),
            x=[1] * len(pr_chars),
            orientation="h",
            marker_color="#B4413C",
            text=pr_chars,
            textposition="inside",
            textfont=dict(color="white", size=10),
            showlegend=False,
            cliponaxis=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            y=list(range(len(gd_chars))),
            x=[1] * len(gd_chars),
            orientation="h",
            marker_color="#1FB8CD",
            text=gd_chars,
            textposition="inside",
            textfont=dict(color="white", size=10),
            showlegend=False,
            cliponaxis=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(title="Pattern Recognition vs Discovery", showlegend=False)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    return fig

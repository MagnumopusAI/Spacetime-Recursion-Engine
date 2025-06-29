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


def construct_validation_lattice() -> go.Figure:
    """Return a diagram of the meta-cognitive validation framework.

    This routine organizes reasoning checks much like nodes on a
    physical lattice.  The connections mirror how field lines link
    interacting particles, providing an intuitive view of how each
    component supports consistent decision making.
    """

    compute_lambda_4_eigenmode(sigma=0.1, tau=0.2)

    data = {
        "framework_components": {
            "main_branches": [
                "Confidence Assessment",
                "Pattern vs Discovery Detection",
                "Reasoning Validation",
                "Domain Boundary Recognition",
            ],
            "confidence_assessment": [
                "Self-calibration",
                "Uncertainty quantification",
                "Fact-level confidence",
                "Temperature scaling",
            ],
            "pattern_detection": [
                "Spurious correlation detection",
                "Novelty assessment",
                "Domain extrapolation flags",
                "Convex hull analysis",
            ],
            "reasoning_validation": [
                "Logical consistency checks",
                "Assumption questioning",
                "Alternative viewpoint consideration",
                "Chain-of-thought verification",
            ],
            "domain_boundary": [
                "Training distribution bounds",
                "Extrapolation warnings",
                "Validity domain mapping",
                "Out-of-distribution detection",
            ],
        }
    }

    colors = ["#1FB8CD", "#FFC185", "#ECEBD5", "#5D878F"]

    fig = go.Figure()

    root_pos = (0, 8)
    branch_positions = [(-6, 5), (-2, 5), (2, 5), (6, 5)]
    sub_positions = [
        [(-6, 3), (-6, 2), (-6, 1), (-6, 0)],
        [(-2, 3), (-2, 2), (-2, 1), (-2, 0)],
        [(2, 3), (2, 2), (2, 1), (2, 0)],
        [(6, 3), (6, 2), (6, 1), (6, 0)],
    ]

    fig.add_trace(
        go.Scatter(
            x=[root_pos[0]],
            y=[root_pos[1]],
            mode="markers+text",
            marker=dict(size=35, color="#13343B"),
            text=["Meta-Cognitive<br>Validation<br>Framework"],
            textposition="middle center",
            textfont=dict(size=16, color="white"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    def abbreviate(text: str) -> str:
        mapping = {
            "Confidence Assessment": "Confidence<br>Assessment",
            "Pattern vs Discovery Detection": "Pattern vs<br>Discovery",
            "Reasoning Validation": "Reasoning<br>Validation",
            "Domain Boundary Recognition": "Domain<br>Boundary",
            "Self-calibration": "Self-<br>Calibration",
            "Uncertainty quantification": "Uncertainty<br>Quantify",
            "Fact-level confidence": "Fact-Level<br>Confidence",
            "Temperature scaling": "Temperature<br>Scaling",
            "Spurious correlation detection": "Spurious<br>Correlation",
            "Novelty assessment": "Novelty<br>Assessment",
            "Domain extrapolation flags": "Domain<br>Extrap Flags",
            "Convex hull analysis": "Convex Hull<br>Analysis",
            "Logical consistency checks": "Logic<br>Consistency",
            "Assumption questioning": "Assumption<br>Questioning",
            "Alternative viewpoint consideration": "Alternative<br>Viewpoints",
            "Chain-of-thought verification": "Chain-of-<br>Thought",
            "Training distribution bounds": "Training<br>Bounds",
            "Extrapolation warnings": "Extrap<br>Warnings",
            "Validity domain mapping": "Validity<br>Mapping",
            "Out-of-distribution detection": "OOD<br>Detection",
        }
        return mapping.get(text, text)

    main_branches = data["framework_components"]["main_branches"]
    sub_components = [
        data["framework_components"]["confidence_assessment"],
        data["framework_components"]["pattern_detection"],
        data["framework_components"]["reasoning_validation"],
        data["framework_components"]["domain_boundary"],
    ]

    for i, (branch, pos, color) in enumerate(zip(main_branches, branch_positions, colors)):
        fig.add_trace(
            go.Scatter(
                x=[pos[0]],
                y=[pos[1]],
                mode="markers+text",
                marker=dict(size=30, color=color, line=dict(color="white", width=2)),
                text=[abbreviate(branch)],
                textposition="middle center",
                textfont=dict(size=12, color="black"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[root_pos[0], pos[0]],
                y=[root_pos[1], pos[1]],
                mode="lines",
                line=dict(color="#13343B", width=3),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        for sub_comp, sub_pos in zip(sub_components[i], sub_positions[i]):
            fig.add_trace(
                go.Scatter(
                    x=[sub_pos[0]],
                    y=[sub_pos[1]],
                    mode="markers+text",
                    marker=dict(size=20, color=color, line=dict(color="white", width=2)),
                    text=[abbreviate(sub_comp)],
                    textposition="middle center",
                    textfont=dict(size=11, color="black"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[pos[0], sub_pos[0]],
                    y=[pos[1], sub_pos[1]],
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    interdependencies = [
        ((-6, 1.5), (2, 2.5)),
        ((-2, 2.5), (6, 1.5)),
    ]

    for start, end in interdependencies:
        fig.add_trace(
            go.Scatter(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                mode="lines",
                line=dict(color="#B4413C", width=2, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="Meta-Cognitive Validation Framework",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-8, 8]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-1, 9]),
        plot_bgcolor="white",
        showlegend=False,
    )

    return fig

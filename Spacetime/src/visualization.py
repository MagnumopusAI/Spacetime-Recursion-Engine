import plotly.graph_objects as go

def compute_lambda_4_eigenmode(sigma: float, tau: float):
    """
    Dummy function to enforce PCE constraints symbolically for λ=4 eigenmode.

    This is a placeholder to indicate where
    eigenmode logic would be implemented in a full PCE modeling system.
    """
    # In a full system, here you’d validate or project onto λ=4 eigenspace.
    pass

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

def build_eigenmode_filter_diagram(eigenvalues, selected_mode):
    """Return a simple bar diagram highlighting the selected eigenmode."""
    fig = go.Figure()
    fig.add_bar(x=list(range(len(eigenvalues))), y=eigenvalues, name="modes")
    fig.add_bar(x=[selected_mode], y=[eigenvalues[selected_mode-1]], name="selected",
                marker_color="#B4413C")
    fig.update_layout(title="Eigenmode Filter", barmode="overlay")
    return fig


def pattern_recognition_vs_discovery():
    """Compare pattern recognition and discovery approaches."""
    fig = go.Figure()
    fig.add_bar(x=["Pattern Recognition"], y=[1], name="Recognition")
    fig.add_bar(x=["Discovery"], y=[1], name="Discovery")
    fig.update_layout(title="Pattern Recognition vs Discovery")
    return fig


def construct_validation_lattice():
    """Wrapper returning the meta-cognitive validation diagram."""
    return build_meta_cognitive_validation_figure()

def build_meta_cognitive_validation_figure():
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
                "Temperature",
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

    # Color codes for the main branches
    colors = ["#1FB8CD", "#FFC185", "#ECEBD5", "#5D878F"]

    fig = go.Figure()

    # Position definition
    root_pos = (0, 8)
    branch_positions = [(-6, 5), (-2, 5), (2, 5), (6, 5)]
    sub_positions = [
        [(-6, 3), (-6, 2), (-6, 1), (-6, 0)],
        [(-2, 3), (-2, 2), (-2, 1), (-2, 0)],
        [(2, 3), (2, 2), (2, 1), (2, 0)],
        [(6, 3), (6, 2), (6, 1), (6, 0)],
    ]

    # Draw root node
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

    main_branches = data["framework_components"]["main_branches"]
    sub_components = [
        data["framework_components"]["confidence_assessment"],
        data["framework_components"]["pattern_detection"],
        data["framework_components"]["reasoning_validation"],
        data["framework_components"]["domain_boundary"],
    ]

    # Draw branches and children
    for i, (branch, pos, color) in enumerate(zip(main_branches, branch_positions, colors)):
        # Branch node
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
        # Edge from root to branch
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
        # Sub-branches
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
            # Edge from branch to sub-branch
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

    # Optional: interdependencies (dashed lines)
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

# To render in a Jupyter notebook:
if __name__ == "__main__":
    fig = build_meta_cognitive_validation_figure()
    fig.show()

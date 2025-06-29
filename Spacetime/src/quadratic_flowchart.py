import plotly.graph_objects as go
from pathlib import Path


def quadratic_forms_flowchart(output_path: Path) -> None:
    """Generate a flowchart of quadratic forms in physics.

    The visualization shows how quadratic and eigenmode relations
    in diverse physical theories stem from a common structure,
    much like branches growing from a shared trunk.

    Parameters
    ----------
    output_path:
        Location to save the resulting HTML figure.
    """
    nodes = {
        'General Form': {'x': 0, 'y': 0, 'color': '#1FB8CD', 'size': 25},
        'Damped Osc': {'x': -2, 'y': 1.5, 'color': '#FFC185', 'size': 20},
        'LCL Filter': {'x': 2, 'y': 1.5, 'color': '#ECEBD5', 'size': 20},
        'Plasma Disp': {'x': -2, 'y': -1.5, 'color': '#5D878F', 'size': 20},
        'Stark Effect': {'x': 0, 'y': 2, 'color': '#D2BA4C', 'size': 20},
        'Zeeman Effect': {'x': 2, 'y': -1.5, 'color': '#B4413C', 'size': 20},
        'Eigenmode': {'x': -1, 'y': -0.5, 'color': '#964325', 'size': 15},
        'Real Roots': {'x': -3, 'y': 0, 'color': '#944454', 'size': 15},
        'Critical': {'x': -3, 'y': -0.8, 'color': '#13343B', 'size': 15},
        'Complex': {'x': -3, 'y': 0.8, 'color': '#DB4545', 'size': 15},
        'Mass/Induct': {'x': 1, 'y': -2.5, 'color': '#1FB8CD', 'size': 12},
        'Damping': {'x': 0, 'y': -3, 'color': '#FFC185', 'size': 12},
        'Frequencies': {'x': -1, 'y': -2.5, 'color': '#ECEBD5', 'size': 12}
    }

    connections = [
        ('General Form', 'Damped Osc'),
        ('General Form', 'LCL Filter'),
        ('General Form', 'Plasma Disp'),
        ('General Form', 'Stark Effect'),
        ('General Form', 'Zeeman Effect'),
        ('General Form', 'Eigenmode'),
        ('General Form', 'Real Roots'),
        ('General Form', 'Critical'),
        ('General Form', 'Complex'),
        ('Damped Osc', 'Mass/Induct'),
        ('LCL Filter', 'Mass/Induct'),
        ('Plasma Disp', 'Frequencies'),
        ('Stark Effect', 'Damping'),
        ('Zeeman Effect', 'Frequencies'),
        ('Real Roots', 'Mass/Induct'),
        ('Critical', 'Damping'),
        ('Complex', 'Frequencies')
    ]

    fig = go.Figure()
    for start, end in connections:
        x0, y0 = nodes[start]['x'], nodes[start]['y']
        x1, y1 = nodes[end]['x'], nodes[end]['y']
        fig.add_shape(
            type="line",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color="rgba(128,128,128,0.3)", width=1),
        )
        # arrow positioned 80% along the line
        x_arrow = x0 + 0.8 * (x1 - x0)
        y_arrow = y0 + 0.8 * (y1 - y0)
        dx = x1 - x0
        dy = y1 - y0
        length = (dx**2 + dy**2) ** 0.5
        if length:
            dx_n = dx / length * 0.1
            dy_n = dy / length * 0.1
            fig.add_shape(
                type="path",
                path=f"M{x_arrow-dx_n},{y_arrow-dy_n+0.05} L{x_arrow},{y_arrow} L{x_arrow-dx_n},{y_arrow-dy_n-0.05}",
                line=dict(color="rgba(128,128,128,0.5)", width=2),
            )

    for name, props in nodes.items():
        fig.add_trace(
            go.Scatter(
                x=[props['x']],
                y=[props['y']],
                mode='markers+text',
                text=[name],
                textposition='middle center',
                textfont=dict(size=10, color='white'),
                marker=dict(size=props['size'], color=props['color'], line=dict(width=2, color='white')),
                showlegend=False,
                hovertemplate='%{text}<extra></extra>'
            )
        )

    fig.update_layout(
        title='Quadratic Forms in Physics',
        xaxis=dict(visible=False, range=[-4, 3]),
        yaxis=dict(visible=False, range=[-4, 3]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    output_path = Path(output_path)
    fig.write_html(str(output_path))


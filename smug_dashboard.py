"""
smug_dashboard
=================
An interactive dashboard for visualizing SMUG solver performance.
"""

import json
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# -----------------------------------------------------------------------------
# Data Synthesis and Loading
# -----------------------------------------------------------------------------

def synthesize_benchmark_manifold(results_path: str = "results.csv",
                                  log_path: str = "run_log.jsonl"):
    """Return synthetic benchmark data.

    The function constructs a small experimental manifold of solver runs,
    much like populating a laboratory bench with carefully prepared samples.

    Parameters
    ----------
    results_path:
        File path to write the CSV summary of solver runs.
    log_path:
        File path to write the JSON Lines log of heuristic events.

    Returns
    -------
    tuple of (DataFrame, DataFrame)
        The first element mirrors the CSV summary and the second mirrors the
        JSONL log, allowing the caller to inspect the synthesized field without
        touching disk.
    """
    results_data = {
        "instance": ["inst1.cnf"] * 3
        + ["inst2.cnf"] * 3
        + ["inst3.cnf"] * 3
        + ["inst4.cnf"] * 3,
        "solver": ["smug_base", "smug_rewire", "minisat"] * 4,
        "runtime": [10.5, 8.2, 15.1, 25.0, 28.1, 22.3, 5.1, 5.0, 9.8, 100.0, 55.2, 100.0],
        "result": [
            "SAT",
            "SAT",
            "SAT",
            "UNSAT",
            "UNSAT",
            "UNSAT",
            "SAT",
            "SAT",
            "SAT",
            "TIMEOUT",
            "SAT",
            "TIMEOUT",
        ],
    }
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(results_path, index=False)

    log_data = [
        {"ts": 1.0, "event": "decision", "level": 1, "var": 10, "v_score": 0.9},
        {"ts": 1.1, "event": "propagate", "level": 1, "var": 5, "val": True, "antecedent": "C10"},
        {"ts": 1.2, "event": "conflict", "level": 1, "clause_id": "C15"},
        {"ts": 1.3, "event": "learn", "level": 1, "clause_id": "L1", "size": 4, "lbd": 2},
        {"ts": 1.4, "event": "restart", "count": 1},
        {"ts": 2.0, "event": "decision", "level": 1, "var": 25, "v_score": 0.85},
        {"ts": 2.2, "event": "conflict", "level": 1, "clause_id": "C88"},
        {"ts": 2.3, "event": "learn", "level": 1, "clause_id": "L2", "size": 3, "lbd": 1},
    ]
    with open(log_path, "w") as f:
        for entry in log_data:
            f.write(json.dumps(entry) + "\n")
    df_log = pd.DataFrame(log_data)
    return df_results, df_log


def load_phase_space_records(results_path: str = "results.csv",
                             log_path: str = "run_log.jsonl"):
    """Load benchmark results and solver trajectories from disk.

    Retrieving these records is akin to charting coordinates in phase space;
    each point contributes to a broader understanding of the solver's motion.
    """
    try:
        df_results = pd.read_csv(results_path)
    except FileNotFoundError:
        df_results = pd.DataFrame()

    try:
        with open(log_path, "r") as f:
            df_log = pd.DataFrame(json.loads(line) for line in f)
    except FileNotFoundError:
        df_log = pd.DataFrame()

    return df_results, df_log


# -----------------------------------------------------------------------------
# Analysis Functions
# -----------------------------------------------------------------------------

def compute_regression_lagrangian(df_results: pd.DataFrame, timeout: float = 100.0) -> pd.DataFrame:
    """Compute PAR-2 scores while honoring the Preservation Constraint Equation.

    Timeouts are treated as double the available time quantum, preserving fairness
    in much the same way a conservation law guards a system's total energy.
    """
    if df_results.empty:
        return pd.DataFrame(columns=["solver", "Solved", "PAR-2"])

    penalty = 2 * timeout
    summary = df_results.groupby("solver").apply(
        lambda x: pd.Series(
            {
                "Solved": (x["result"] != "TIMEOUT").sum(),
                "PAR-2": x.apply(
                    lambda row: row["runtime"]
                    if row["result"] != "TIMEOUT"
                    else penalty,
                    axis=1,
                ).sum(),
            }
        )
    ).reset_index()
    return summary


def render_cactus_geometry(df_results: pd.DataFrame, solver1: str, solver2: str):
    """Create a runtime comparison plot.

    The scatter plot mirrors two particles racing through spacetime, with the
    diagonal representing equal velocity world-lines.
    """
    if df_results.empty or not solver1 or not solver2:
        return go.Figure().update_layout(
            template="plotly_dark", title="Please select two solvers."
        )

    df1 = df_results[df_results["solver"] == solver1]
    df2 = df_results[df_results["solver"] == solver2]
    merged = pd.merge(df1, df2, on="instance", suffixes=("_1", "_2"))

    fig = px.scatter(
        merged,
        x="runtime_1",
        y="runtime_2",
        hover_data=["instance"],
        title=f"Runtime Comparison: {solver1} vs {solver2}",
    )
    max_val = merged[["runtime_1", "runtime_2"]].max().max()
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_val,
        y1=max_val,
        line=dict(color="gray", dash="dash"),
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title=f"{solver1} Runtime (s)",
        yaxis_title=f"{solver2} Runtime (s)",
        font=dict(color="white"),
    )
    return fig


def trace_heuristic_orbit(df_log: pd.DataFrame):
    """Visualize the evolution of learned clause quality.

    The trajectory resembles tracking a satellite's orbital decay where LBD is
    analogous to altitude relative to problem complexity.
    """
    if df_log.empty:
        return go.Figure().update_layout(
            template="plotly_dark", title="No trajectory log data loaded."
        )

    df_learn = df_log[df_log["event"] == "learn"].copy()
    df_learn["time_group"] = (df_learn["ts"] // 0.5) * 0.5
    avg_lbd = df_learn.groupby("time_group")["lbd"].mean().reset_index()

    fig = px.line(
        avg_lbd,
        x="time_group",
        y="lbd",
        title="Learned Clause Quality (LBD) Over Time",
        markers=True,
    )
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Time (s)",
        yaxis_title="Average LBD",
        font=dict(color="white"),
    )
    return fig


# -----------------------------------------------------------------------------
# Dash Application
# -----------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"
    ],
)
app.title = "SMUG Solver Dashboard"

# Prepare data for the dashboard
synthesize_benchmark_manifold()
df_results, df_log = load_phase_space_records()

app.layout = html.Div(
    className="bg-gray-900 text-white min-h-screen font-sans p-8",
    children=[
        html.H1(
            "SMUG Solver: Benchmarking War Room",
            className="text-4xl font-bold mb-2 text-cyan-400",
        ),
        html.P(
            "Analysis and Intelligence Dashboard",
            className="text-lg text-gray-400 mb-8",
        ),
        # Section 1: Regression Test Dashboard
        html.Div(
            className="bg-gray-800 p-6 rounded-lg shadow-lg mb-8",
            children=[
                html.H2(
                    "Regression Test Summary",
                    className="text-2xl font-semibold mb-4 text-cyan-300",
                ),
                html.Div(id="regression-summary-table", className="overflow-x-auto"),
            ],
        ),
        # Section 2: Comparative Analysis
        html.Div(
            className="bg-gray-800 p-6 rounded-lg shadow-lg mb-8",
            children=[
                html.H2(
                    "Comparative Performance Analysis",
                    className="text-2xl font-semibold mb-4 text-cyan-300",
                ),
                html.Div(
                    className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4",
                    children=[
                        dcc.Dropdown(
                            id="solver1-dropdown",
                            options=[
                                {"label": s, "value": s} for s in df_results["solver"].unique()
                            ],
                            value=
                                df_results["solver"].unique()[0]
                                if not df_results.empty
                                else None,
                            className="text-black",
                        ),
                        dcc.Dropdown(
                            id="solver2-dropdown",
                            options=[
                                {"label": s, "value": s} for s in df_results["solver"].unique()
                            ],
                            value=
                                df_results["solver"].unique()[1]
                                if len(df_results["solver"].unique()) > 1
                                else None,
                            className="text-black",
                        ),
                    ],
                ),
                dcc.Graph(id="cactus-plot", className="rounded-lg"),
            ],
        ),
        # Section 3: Heuristic Trajectory Visualization
        html.Div(
            className="bg-gray-800 p-6 rounded-lg shadow-lg",
            children=[
                html.H2(
                    "Heuristic Trajectory Visualization",
                    className="text-2xl font-semibold mb-4 text-cyan-300",
                ),
                html.P(
                    "Visualizing internal solver metrics from a single run log.",
                    className="text-gray-400 mb-4",
                ),
                dcc.Graph(id="trajectory-plot", className="rounded-lg"),
            ],
        ),
    ],
)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------


@app.callback(Output("regression-summary-table", "children"), Input("solver1-dropdown", "value"))
def update_regression_table(_):
    """Update regression summary table."""
    summary = compute_regression_lagrangian(df_results)
    if summary.empty:
        return html.P("No results data loaded.")

    return html.Table(
        className="min-w-full divide-y divide-gray-700",
        children=[
            html.Thead(
                html.Tr(
                    [
                        html.Th(
                            "Solver",
                            className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider",
                        ),
                        html.Th(
                            "Instances Solved",
                            className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider",
                        ),
                        html.Th(
                            "PAR-2 Score",
                            className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider",
                        ),
                    ]
                )
            ),
            html.Tbody(
                className="bg-gray-800 divide-y divide-gray-700",
                children=[
                    html.Tr(
                        [
                            html.Td(row["solver"], className="px-6 py-4 whitespace-nowrap"),
                            html.Td(row["Solved"], className="px-6 py-4 whitespace-nowrap"),
                            html.Td(f"{row['PAR-2']:.2f}", className="px-6 py-4 whitespace-nowrap"),
                        ]
                    )
                    for _, row in summary.iterrows()
                ],
            ),
        ],
    )


@app.callback(
    Output("cactus-plot", "figure"),
    [Input("solver1-dropdown", "value"), Input("solver2-dropdown", "value")],
)
def update_cactus_plot(solver1, solver2):
    """Refresh runtime comparison plot based on dropdown selections."""
    return render_cactus_geometry(df_results, solver1, solver2)


@app.callback(Output("trajectory-plot", "figure"), Input("solver1-dropdown", "value"))
def update_trajectory_plot(_):
    """Refresh the trajectory plot."""
    return trace_heuristic_orbit(df_log)


if __name__ == "__main__":
    app.run_server(debug=True)

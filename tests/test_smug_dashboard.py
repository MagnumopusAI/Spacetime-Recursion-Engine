import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:  # pragma: no cover
    from smug_dashboard import (
        synthesize_benchmark_manifold,
        compute_regression_lagrangian,
    )
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("dash is not installed", allow_module_level=True)


def test_par_two_preservation(tmp_path):
    """Ensure PAR-2 obeys a preservation rule akin to energy conservation."""
    results_path = tmp_path / "results.csv"
    log_path = tmp_path / "run_log.jsonl"
    df_results, _ = synthesize_benchmark_manifold(results_path, log_path)
    summary = compute_regression_lagrangian(df_results, timeout=100.0)
    table = summary.set_index("solver")

    assert table.loc["smug_rewire", "Solved"] == 4
    assert table.loc["smug_rewire", "PAR-2"] == pytest.approx(96.5)
    assert table.loc["smug_base", "PAR-2"] == pytest.approx(240.6)
    assert table.loc["minisat", "PAR-2"] == pytest.approx(247.2)

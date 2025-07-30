import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.three_sat_solver import solve_three_sat


@pytest.fixture
def example_clauses():
    return [
        (1, -2, 3),
        (-1, 2, -3),
        (1, 2, -3),
    ]


def test_solve_three_sat_example(example_clauses):
    assignment = solve_three_sat(example_clauses)
    assert assignment is not None
    assert all(
        any(assignment[abs(l)] if l > 0 else not assignment[abs(l)] for l in c)
        for c in example_clauses
    )


def test_unsatisfiable_instance():
    clauses = [(1, 1, 1), (-1, -1, -1)]
    assert solve_three_sat(clauses) is None


def test_invalid_clauses():
    with pytest.raises(ValueError):
        solve_three_sat([(0, 1, -2)])

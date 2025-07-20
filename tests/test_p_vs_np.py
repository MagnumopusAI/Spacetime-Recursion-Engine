import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.millennium.p_vs_np import pce_solve_for_tau, resolve_p_vs_np


def test_pce_solve_for_tau_equation():
    sol1, sol2 = pce_solve_for_tau(1.0)
    # Check that both roots satisfy 2 τ^2 + 3 τ − 2 = 0
    assert abs(2 * sol1**2 + 3 * sol1 - 2) <= 1e-6
    assert abs(2 * sol2**2 + 3 * sol2 - 2) <= 1e-6


def test_resolve_p_vs_np_formalist():
    clauses = [(1, -2, 3), (-1, 2, -3)]
    assignment = resolve_p_vs_np(clauses, observer_mode="formalist")
    assert assignment is not None
    # Each clause must be satisfied by the returned boolean assignment
    assert all(
        any(
            (assignment[abs(lit)] if lit > 0 else not assignment[abs(lit)])
            for lit in clause
        )
        for clause in clauses
    )


def test_resolve_p_vs_np_smug():
    clauses = [(1, -2, 3), (-1, 2, -3)]
    assignment = resolve_p_vs_np(clauses, observer_mode="smug")
    assert assignment is not None
    # Again ensure every clause is satisfied
    assert all(
        any(
            (assignment[abs(lit)] if lit > 0 else not assignment[abs(lit)])
            for lit in clause
        )
        for clause in clauses
    )


def test_resolve_p_vs_np_invalid_mode():
    clauses = [(1, -2, 3)]
    try:
        resolve_p_vs_np(clauses, observer_mode="invalid")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for invalid observer_mode"

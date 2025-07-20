import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.master_toolkit import (
    resolve_p_vs_np,
    pce_equation_check,
    pce_solve_for_tau,
    get_curve_equation,
)
from sympy import Eq, symbols


def test_resolve_p_vs_np_formalist():
    clauses = [(1, 1, 1)]
    result = resolve_p_vs_np(clauses, observer_mode="formalist")
    assert result is not None
    assert result[1]


def test_resolve_p_vs_np_smug():
    clauses = [(-1, 2, 3), (1, -2, 3)]
    result = resolve_p_vs_np(clauses, observer_mode="smug")
    assert result is not None
    for clause in clauses:
        assert any(result[abs(l)] if l > 0 else not result[abs(l)] for l in clause)


def test_invalid_mode_raises():
    try:
        resolve_p_vs_np([(1, 1, 1)], observer_mode="unknown")
    except ValueError:
        assert True
    else:
        assert False


def test_pce_equation_check_true():
    sigma = 1.0
    tau = pce_solve_for_tau(sigma)
    assert pce_equation_check(sigma, tau)


def test_pce_equation_check_false():
    assert not pce_equation_check(0.5, 1.0)


def test_get_curve_equation():
    x, y = symbols("x y")
    eq = get_curve_equation(1, 2)
    assert eq == Eq(y ** 2, x ** 3 + 1 * x + 2)


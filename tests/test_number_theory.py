import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.number_theory import (
    radical,
    check_common_prime_factor,
    pce_solve_for_tau,
    define_elliptic_curve,
    hodge_star_operator,
    get_l_function_rank,
)


def test_radical_basic():
    assert radical(18) == 6
    assert radical(7) == 7
    assert radical(0) == 0


def test_common_prime_factor():
    assert check_common_prime_factor(8, 12, 18)
    assert not check_common_prime_factor(3, 4, 5)


def test_pce_solve_for_tau():
    sigma = 1.0
    solutions = pce_solve_for_tau(sigma)
    disc = (9 + 16 * sigma**2) ** 0.5
    expected = {(-3 + disc) / 4, (-3 - disc) / 4}
    assert set(round(s, 12) for s in solutions) == set(round(e, 12) for e in expected)


def test_define_elliptic_curve():
    curve = define_elliptic_curve(2, 3)
    x, y = curve.free_symbols
    assert str(curve) == f'Eq({y}**2, {x}**3 + 2*{x} + 3)'


def test_hodge_star_operator():
    assert hodge_star_operator(1, 4) == 3
    assert hodge_star_operator(5, 4) == 'Invalid: k-form must be 0 <= k <= 4.'


def test_l_function_rank():
    assert get_l_function_rank((1, -2)) == 3

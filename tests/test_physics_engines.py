import sys
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.physics_engines import (
    radical,
    check_common_prime_factor,
    pce_solve_for_tau,
    define_elliptic_curve,
    hodge_star_operator,
    navier_stokes_stability_check,
    calculate_smug_mass_gap,
)


def test_radical():
    assert radical(12) == 6
    assert radical(0) == 0


def test_common_prime_factor():
    assert check_common_prime_factor(8, 12, 16)
    assert not check_common_prime_factor(8, 9, 25)


def test_pce_solve_for_tau():
    sigma = 0.5
    tau1, tau2 = pce_solve_for_tau(sigma)
    eq = lambda t: 2 * t**2 + 3 * t - 2 * sigma**2
    assert math.isclose(eq(tau1), 0.0, abs_tol=1e-9)
    assert math.isclose(eq(tau2), 0.0, abs_tol=1e-9)


def test_define_elliptic_curve():
    eq = define_elliptic_curve(1, 1)
    assert str(eq) == 'Eq(y**2, x**3 + x + 1)'


def test_hodge_star_operator():
    assert hodge_star_operator(1, 4) == 3


def test_navier_stokes_stability_check():
    stable = navier_stokes_stability_check(1.0, 1.0)
    unstable = navier_stokes_stability_check(6000.0, 1.0)
    assert 'SMOOTH' in stable
    assert 'SINGULARITY' in unstable


def test_calculate_smug_mass_gap():
    gap = calculate_smug_mass_gap(1.0)
    assert gap > 0.0

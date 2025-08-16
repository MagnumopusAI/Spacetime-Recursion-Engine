import math
import sys
from pathlib import Path
import unittest
from sympy import Eq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.pce_symbolic_verifier import (
    solve_pce_symbolically,
    pce_expression,
    is_pce_preserved,
    derive_fold_surface_balance,
    coarse_grained_delta_g,
    estimate_gamma,
    E_FOLD,
    E_SURFACE,
    GAMMA,
)


class TestSymbolicPCEVerifier(unittest.TestCase):
    """Unit tests for symbolic PCE verification utilities."""

    def test_solve_pce_symbolically(self):
        sigma = 1.5
        taus = solve_pce_symbolically(sigma)
        for tau in taus:
            self.assertAlmostEqual(pce_expression(sigma, float(tau)), 0.0, places=9)

    def test_rotation_breaks_invariance(self):
        sigma = 1.5
        tau = float(solve_pce_symbolically(sigma)[1])
        upsilon = tau
        self.assertTrue(is_pce_preserved(sigma, tau, upsilon, 0.0))
        self.assertFalse(is_pce_preserved(sigma, tau, upsilon, math.pi / 4))

    def test_derive_fold_surface_balance(self):
        equation = derive_fold_surface_balance()
        expected = Eq(E_FOLD**2 - GAMMA**2 * E_SURFACE**2, 0)
        self.assertEqual(equation, expected)

    def test_coarse_grained_delta_g_samples(self):
        samples = {
            "proteinA": {
                "e_fold": -60.0,
                "e_surface": -50.0,
                "gamma": 1.1,
                "delta_g": -5.0,
            },
            "proteinB": {
                "e_fold": -45.0,
                "e_surface": -30.0,
                "gamma": 1.3,
                "delta_g": -6.0,
            },
        }
        for sample in samples.values():
            calc = coarse_grained_delta_g(
                sample["e_fold"], sample["e_surface"], sample["gamma"]
            )
            self.assertAlmostEqual(calc, sample["delta_g"], places=6)
            gamma_est = estimate_gamma(
                sample["delta_g"], sample["e_fold"], sample["e_surface"]
            )
            self.assertAlmostEqual(gamma_est, sample["gamma"], places=6)


if __name__ == "__main__":
    unittest.main()

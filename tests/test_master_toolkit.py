import sys
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.master_toolkit import (
    resolve_p_vs_np,
    radical,
    check_common_prime_factor,
    pce_solve_for_tau,
    define_elliptic_curve,
    hodge_star_operator,
    navier_stokes_stability_check,
    calculate_smug_mass_gap,
    pce_equation_check,
    get_curve_equation,
)
from sympy import Eq, symbols


class TestResolvePvNP(unittest.TestCase):
    def test_formalist_solves_simple(self):
        clauses = [(1, 1, 1)]
        result = resolve_p_vs_np(clauses, observer_mode="formalist")
        self.assertIsNotNone(result)
        self.assertTrue(result[1])

    def test_smug_solves_simple(self):
        clauses = [(-1, 2, 3), (1, -2, 3)]
        result = resolve_p_vs_np(clauses, observer_mode="smug")
        self.assertIsNotNone(result)
        for clause in clauses:
            self.assertTrue(
                any((result[abs(l)] if l > 0 else not result[abs(l)]) for l in clause)
            )

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            resolve_p_vs_np([(1, 1, 1)], observer_mode="unknown")


class TestToolkitUtils(unittest.TestCase):
    def test_radical(self):
        self.assertEqual(radical(60), 30)
        self.assertEqual(radical(0), 0)

    def test_check_common_prime_factor(self):
        self.assertTrue(check_common_prime_factor(15, 25, 35))
        self.assertFalse(check_common_prime_factor(2, 3, 4))

    def test_pce_solve_for_tau(self):
        solutions = pce_solve_for_tau(1.0)
        self.assertIn(0.5, solutions)
        self.assertIn(-2.0, solutions)

    def test_define_elliptic_curve(self):
        x, y = symbols('x y')
        expected = Eq(y**2, x**3 + x + 1)
        self.assertEqual(define_elliptic_curve(1, 1), expected)

    def test_hodge_star_operator(self):
        self.assertEqual(hodge_star_operator(1, 4), 3)
        with self.assertRaises(ValueError):
            hodge_star_operator(5, 4)

    def test_navier_stokes_stability_check(self):
        self.assertEqual(
            navier_stokes_stability_check(1.0, 0.0),
            "Invalid fluid: Viscosity must be positive."
        )
        self.assertIn(
            "SINGULARITY", navier_stokes_stability_check(6000.0, 1.0)
        )
        self.assertIn(
            "SMOOTH FLOW", navier_stokes_stability_check(10.0, 5.0)
        )

    def test_calculate_smug_mass_gap(self):
        result = calculate_smug_mass_gap(0.5)
        expected = math.exp(-1.0 / (0.5 ** 2))
        self.assertAlmostEqual(result, expected, places=12)
        self.assertEqual(calculate_smug_mass_gap(0.0), float('inf'))


class TestPCEFunctions(unittest.TestCase):
    def test_pce_equation_true(self):
        sigma = 1.0
        for tau in pce_solve_for_tau(sigma):
            self.assertTrue(pce_equation_check(sigma, tau))

    def test_pce_equation_false(self):
        self.assertFalse(pce_equation_check(0.5, 1.0))


class TestGetCurveEquation(unittest.TestCase):
    def test_returns_correct_equation(self):
        x, y = symbols("x y")
        eq = get_curve_equation(1, 2)
        self.assertEqual(eq, Eq(y**2, x**3 + 1 * x + 2))


if __name__ == "__main__":
    unittest.main()

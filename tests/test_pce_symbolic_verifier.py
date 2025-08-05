import math
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.pce_symbolic_verifier import (
    solve_pce_symbolically,
    pce_expression,
    is_pce_preserved,
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


if __name__ == "__main__":
    unittest.main()

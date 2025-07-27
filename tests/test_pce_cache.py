import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.pce_cache import (
    cached_pce_solve_for_tau,
    batch_cached_pce,
    PCE_EXPRESSION,
)
from sympy import symbols


def test_cached_pce_solve_for_tau_values():
    sigma = 0.5
    taus = cached_pce_solve_for_tau(sigma)
    tau_sym, sigma_sym = symbols("tau sigma")
    eq = PCE_EXPRESSION.subs(sigma_sym, sigma)
    # verify each solution satisfies the equation
    for tau in taus:
        assert abs(eq.subs(tau_sym, tau)) < 1e-9


def test_batch_cached_pce_shape():
    sigmas = [0.5, 1.5, 2.5]
    solutions = batch_cached_pce(sigmas)
    assert solutions.shape == (3, 2)
    # ensure caching works by calling again
    solutions2 = batch_cached_pce(sigmas)
    assert np.allclose(solutions, solutions2)

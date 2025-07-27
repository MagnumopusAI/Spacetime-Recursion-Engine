import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.pce_vectorized import solve_tau_cached, batch_solve_tau


def test_solve_tau_cached_basic():
    sigma = 0.5
    taus = solve_tau_cached(sigma)
    expected = [(-3 + (9 + 16 * sigma ** 2) ** 0.5) / 4,
                (-3 - (9 + 16 * sigma ** 2) ** 0.5) / 4]
    for tau in taus:
        residual = 2 * (tau.real ** 2) + 3 * tau.real - 2 * sigma ** 2
        assert abs(residual) < 0.1


def test_batch_solve_tau_shape_and_values():
    sigmas = [0.5, 1.5]
    matrix = batch_solve_tau(sigmas)
    assert matrix.shape == (2, 2)
    arr = matrix.toarray()
    for sigma, tau_row in zip(sigmas, arr):
        for tau in tau_row:
            residual = 2 * (tau.real ** 2) + 3 * tau.real - 2 * sigma ** 2
            assert abs(residual) < 0.1

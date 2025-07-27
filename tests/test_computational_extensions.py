import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.computational_extensions import (
    optimized_pce_solve_for_tau,
    batch_pce,
)


def test_optimized_pce_single_solution_accuracy():
    np.random.seed(0)
    sigma = 0.5
    taus = optimized_pce_solve_for_tau(sigma)
    for tau in taus:
        residual = np.real(2 * tau**2 + 3 * tau - 2 * sigma**2)
        assert abs(residual) < 0.14


def test_batch_pce_vectorization():
    np.random.seed(0)
    sigmas = [0.5, 1.5, np.sqrt(11), np.sqrt(13)]
    results = batch_pce(sigmas)
    assert results.shape == (len(sigmas), 2)
    for sigma, row in zip(sigmas, results):
        for tau in row:
            residual = np.real(2 * tau**2 + 3 * tau - 2 * sigma**2)
            assert abs(residual) < 0.14

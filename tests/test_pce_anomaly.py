import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.pce_anomaly import (
    solve_anomaly_tau_cached,
    batch_anomaly_detection,
    PCE_ANOMALY,
)
from sympy import symbols


def test_solve_anomaly_tau_cached_values():
    sigma = 0.5
    upsilon = np.pi
    taus, score = solve_anomaly_tau_cached(sigma, upsilon)
    tau_sym, sigma_sym, upsilon_sym = symbols('tau sigma upsilon')
    eq = PCE_ANOMALY.subs({sigma_sym: sigma, upsilon_sym: upsilon})
    for tau in taus:
        residual = eq.subs(tau_sym, tau)
        assert abs(complex(residual)) < 0.1

    # caching should return identical results
    taus2, score2 = solve_anomaly_tau_cached(sigma, upsilon)
    assert np.allclose(taus, taus2)
    assert score2 == score


def test_batch_anomaly_detection_structure():
    np.random.seed(0)
    data_points = [np.random.normal(0, s, 100) for s in (0.5, 1.0)]
    results = batch_anomaly_detection(data_points, np.pi)
    assert len(results) == 2
    for (taus, score), data in zip(results, data_points):
        assert len(taus) == 2
        sigma = float(np.std(data))
        tau_sym, sigma_sym, upsilon_sym = symbols('tau sigma upsilon')
        eq = PCE_ANOMALY.subs({sigma_sym: sigma, upsilon_sym: np.pi})
        residual = eq.subs(tau_sym, taus[0])
        assert abs(complex(residual)) < 0.1
        assert isinstance(score, float) and score >= 0

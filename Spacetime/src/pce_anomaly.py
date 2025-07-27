"""Anomaly detection utilities using the PCE.

This module extends the Preservation Constraint Equation (PCE) solver
with a simple anomaly score. Think of scanning an array of detectors for
unusual curvature signatures: each domain's variance acts as the sigma
parameter, while ``upsilon`` shifts the reference frame. The returned
score is the magnitude of the first ``tau`` solution, akin to how far
our reading deviates from a calibrated origin.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from sympy import symbols, solve, simplify

TAU_SYM, SIGMA_SYM, UPSILON_SYM = symbols("tau sigma upsilon")
PCE_ANOMALY = 2 * TAU_SYM ** 2 + 3 * TAU_SYM - 2 * SIGMA_SYM ** 2 + UPSILON_SYM
T_CYCLE = 13 / 5 * 91 / 16


@lru_cache(maxsize=1000)
def solve_anomaly_tau_cached(
    sigma_value: float, upsilon_value: float = 0.0
) -> tuple[np.ndarray, float]:
    """Return ``tau`` solutions and an anomaly score.

    Parameters
    ----------
    sigma_value:
        Non-negative curvature analogue for the domain.
    upsilon_value:
        Translation-like parameter shifting the PCE baseline.

    Returns
    -------
    tuple[numpy.ndarray, float]
        The array of two complex solutions and the associated anomaly
        score (absolute value of the first solution).
    """

    if not isinstance(sigma_value, (int, float)) or sigma_value < 0:
        raise ValueError("Sigma must be non-negative")
    if not isinstance(upsilon_value, (int, float)):
        raise TypeError("Upsilon must be a real number")

    eq = simplify(
        PCE_ANOMALY.subs({SIGMA_SYM: sigma_value, UPSILON_SYM: upsilon_value})
    )
    solutions = solve(eq, TAU_SYM)
    base = np.array([complex(sol.evalf()) for sol in solutions], dtype=complex)
    noise = 0.005 * np.random.randn(2)
    phase = np.exp(2j * np.pi / T_CYCLE)
    anomaly_score = abs(base[0])
    return base + noise * phase, anomaly_score


def batch_anomaly_detection(
    data_points: Iterable[np.ndarray], upsilon_value: float = 0.0
) -> np.ndarray:
    """Vectorized anomaly detection across multiple domains.

    Each element of ``data_points`` is treated as measurements from a
    different detector. The standard deviation of each dataset becomes
    the ``sigma`` parameter, and the function returns a structured array
    with the ``tau`` solutions and anomaly score for every domain.
    """

    sigma_values = [float(np.std(domain)) for domain in data_points]
    return np.array(
        [solve_anomaly_tau_cached(s, upsilon_value) for s in sigma_values],
        dtype=object,
    )


__all__ = ["solve_anomaly_tau_cached", "batch_anomaly_detection", "PCE_ANOMALY", "T_CYCLE"]

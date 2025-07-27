"""Enhanced Preservation Constraint solver utilities.

This module implements a vectorized solver for the simplified
Preservation Constraint Equation (PCE)::

    2*tau**2 + 3*tau - 2*sigma**2 = 0

The routines mimic how experimental physicists scan across many
curvature values ``sigma`` to find stable torsion states ``tau``.  A
small complex perturbation is introduced to emulate measurement noise.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from sympy import symbols, solve, simplify
from scipy.sparse import csr_matrix


TAU_SYM, SIGMA_SYM = symbols("tau sigma")
PCE_QUADRATIC = 2 * TAU_SYM ** 2 + 3 * TAU_SYM - 2 * SIGMA_SYM ** 2
T_CYCLE = 13 / 5 * 91 / 16  # ~14.79 seconds


@lru_cache(maxsize=1000)
def solve_tau_cached(sigma_value: float) -> np.ndarray:
    """Return ``tau`` solutions for a given ``sigma`` with caching.

    Parameters
    ----------
    sigma_value:
        Non-negative curvature analogue.

    Returns
    -------
    numpy.ndarray
        Complex array containing two solutions with a small phase
        perturbation.  The cache acts like an experimental logbook,
        recalling previously computed states.
    """

    if not isinstance(sigma_value, (int, float)):
        raise TypeError("Sigma must be a non-negative number")
    if sigma_value < 0:
        raise ValueError("Sigma must be a non-negative number")

    simplified = simplify(PCE_QUADRATIC.subs(SIGMA_SYM, sigma_value))
    solutions = solve(simplified, TAU_SYM)
    base = np.array([float(sol.evalf()) for sol in solutions], dtype=float)
    noise = 0.005 * np.random.randn(2)
    phase = np.exp(2j * np.pi / T_CYCLE)
    return base + noise * phase


def batch_solve_tau(sigma_values: Iterable[float]) -> csr_matrix:
    """Vectorized PCE solving over many ``sigma`` values.

    The resulting sparse matrix mimics a thin slice through the
    ``sigma``-``tau`` phase space, storing the two torsion solutions for
    each input ``sigma``.
    """

    tau_array = np.array([solve_tau_cached(float(s)) for s in sigma_values])
    return csr_matrix(tau_array)


__all__ = ["solve_tau_cached", "batch_solve_tau"]

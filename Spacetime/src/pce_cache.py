"""Advanced helpers for the Preservation Constraint Equation."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from sympy import symbols, solve, simplify

# ---------------------------------------------------------------------------
# Symbolic setup
# ---------------------------------------------------------------------------

_tau, _sigma = symbols("tau sigma")
PCE_EXPRESSION = 2 * _tau**2 + 3 * _tau - 2 * _sigma**2

# The recurrence period used by certain SMUG simulations
T_CYCLE = (13 / 5) * (91 / 16)


@lru_cache(maxsize=256)
def cached_pce_solve_for_tau(sigma_value: float) -> np.ndarray:
    """Return the two ``tau`` solutions for a given ``sigma``.

    The function memoizes results to mimic repeatedly consulting a
    reference table of calibrated measurements. ``sigma_value`` must be
    non-negative, reflecting a physical curvature magnitude.
    """

    if sigma_value < 0:
        raise ValueError("Sigma must be non-negative")

    eq = simplify(PCE_EXPRESSION.subs(_sigma, sigma_value))
    solutions = solve(eq, _tau)
    return np.array([float(sol.evalf()) for sol in solutions])


def batch_cached_pce(sigma_values: Iterable[float]) -> np.ndarray:
    """Vectorized PCE solver for a list of ``sigma`` values.

    Conceptually this performs multiple measurements across a lattice,
    returning an array with shape ``(len(sigma_values), 2)``.
    """

    return np.array([cached_pce_solve_for_tau(float(s)) for s in sigma_values])


__all__ = ["cached_pce_solve_for_tau", "batch_cached_pce", "T_CYCLE"]

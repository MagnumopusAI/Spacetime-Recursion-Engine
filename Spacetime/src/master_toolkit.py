"""Toolkit bridging mathematical invariants and computational heuristics.

Each function illustrates a physical or geometric concept with a direct
analogy to the SMUG framework.  The intent is to keep symbolic clarity
and preserve the PCE structure whenever applicable.
"""

from __future__ import annotations

import math
from typing import Iterable, Dict, List
from sympy import Eq, symbols


# ---------------------------------------------------------------------------
# P vs NP toy model
# ---------------------------------------------------------------------------

def resolve_p_vs_np(
    clauses: Iterable[tuple[int, int, int]], *, observer_mode: str = "formalist"
) -> Dict[int, bool] | None:
    """Return a satisfying assignment for the given 3-CNF formula.

    Parameters
    ----------
    clauses:
        Sequence of clauses ``(x_i, x_j, x_k)`` with literals as signed ints.
    observer_mode:
        ``"formalist"`` performs a straight search while ``"smug"`` adds a
        whimsical label. Invalid modes raise ``ValueError``.

    Returns
    -------
    dict[int, bool] | None
        Mapping of variable index to truth value if satisfiable, else ``None``.
    """

    if observer_mode not in {"formalist", "smug"}:
        raise ValueError(f"Unknown observer_mode: {observer_mode}")

    # determine number of variables
    num_vars = 0
    for clause in clauses:
        num_vars = max(num_vars, *(abs(l) for l in clause))

    # brute force search over 2^n assignments
    for assignment_num in range(1 << num_vars):
        assignment = {
            i + 1: bool(assignment_num & (1 << i)) for i in range(num_vars)
        }
        satisfied = True
        for clause in clauses:
            if not any(
                (assignment[abs(l)] if l > 0 else not assignment[abs(l)])
                for l in clause
            ):
                satisfied = False
                break
        if satisfied:
            return assignment
    return None


# ---------------------------------------------------------------------------
# Number theory helpers
# ---------------------------------------------------------------------------

def radical(n: int) -> int:
    """Return the product of unique prime factors of ``n``.

    ``radical(60)`` behaves like extracting the wooden beams from a
    building: the shape remains, but only the supporting structure is left.
    """

    if n == 0:
        return 0
    n = abs(n)
    result = 1
    factor = 2
    while factor * factor <= n:
        if n % factor == 0:
            result *= factor
            while n % factor == 0:
                n //= factor
        factor += 1
    if n > 1:
        result *= n
    return result


def check_common_prime_factor(a: int, b: int, c: int) -> bool:
    """Return ``True`` if ``a``, ``b`` and ``c`` share a prime factor."""

    from math import gcd

    return gcd(gcd(abs(a), abs(b)), abs(c)) > 1


# ---------------------------------------------------------------------------
# Preservation Constraint utilities
# ---------------------------------------------------------------------------

def pce_solve_for_tau(sigma: float) -> List[float]:
    """Return both solutions of the PCE for ``tau``.

    The equation is ``2*tau**2 + 3*tau - 2*sigma**2 = 0`` and the roots
    capture the torsion compatible with the given curvature analogue.
    """

    disc = 9 + 16 * sigma ** 2
    sqrt_disc = math.sqrt(disc)
    tau1 = (-3 + sqrt_disc) / 4
    tau2 = (-3 - sqrt_disc) / 4
    return [tau1, tau2]


def pce_equation_check(sigma: float, tau: float, *, tol: float = 1e-9) -> bool:
    """Return ``True`` if ``sigma`` and ``tau`` satisfy the PCE within ``tol``."""

    value = -2 * sigma ** 2 + 2 * tau ** 2 + 3 * tau
    return abs(value) <= tol


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def define_elliptic_curve(a: int, b: int):
    """Return ``y**2 = x**3 + a*x + b`` as a symbolic equation."""

    x, y = symbols("x y")
    return Eq(y ** 2, x ** 3 + a * x + b)


def get_curve_equation(a: int, b: int):
    """Helper alias for :func:`define_elliptic_curve`."""

    return define_elliptic_curve(a, b)


# ---------------------------------------------------------------------------
# Differential forms
# ---------------------------------------------------------------------------

def hodge_star_operator(k: int, n: int) -> int:
    """Return the degree of the dual form under the Hodge star in ``n`` dims."""

    if k < 0 or k > n:
        raise ValueError("Form degree k must satisfy 0 <= k <= n")
    return n - k


# ---------------------------------------------------------------------------
# Fluid dynamics sketch
# ---------------------------------------------------------------------------

def navier_stokes_stability_check(reynolds: float, viscosity: float) -> str:
    """Return a stability string based on Reynolds number and viscosity."""

    if viscosity <= 0:
        return "Invalid fluid: Viscosity must be positive."
    if reynolds > 5000:
        return "SINGULARITY detected: turbulent breakdown imminent."
    return "SMOOTH FLOW regime: solution expected to remain regular."


# ---------------------------------------------------------------------------
# SMUG mass gap toy model
# ---------------------------------------------------------------------------

def calculate_smug_mass_gap(coupling: float) -> float:
    """Return an exponential mass gap inspired by asymptotic freedom."""

    if coupling == 0:
        return float("inf")
    return math.exp(-1.0 / (coupling ** 2))


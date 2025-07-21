"""
Toolkit bridging mathematical invariants, preservation‐constraint heuristics,
and geometric/physical toy‐models (SMUG/PCE analogy).
"""

from __future__ import annotations

import math
from itertools import product
from math import gcd
from typing import Iterable, Dict, List

from sympy import Eq, symbols

# ---------------------------------------------------------------------------
# P vs NP toy model
# ---------------------------------------------------------------------------

def resolve_p_vs_np(
    clauses: Iterable[tuple[int, ...]],
    *,
    observer_mode: str = "formalist"
) -> Dict[int, bool] | None:
    """
    Attempt a naive 3‐SAT solve as a metaphor for P vs NP.
    
    Parameters
    ----------
    clauses : iterable of tuples
        Each clause is a sequence of signed ints, e.g. (1, -2,3).
        Positive i means variable i, negative i means ¬i.
    observer_mode : {'formalist', 'smug'}
        - 'formalist': stop at the first satisfying assignment.
        - 'smug':      check all possibilities (semantically identical here,
                       but meant to evoke exhaustive scrutiny).
    
    Returns
    -------
    dict[int,bool] | None
        A satisfying assignment mapping var→bool, or None if unsatisfiable.
    """
    variables = sorted({abs(l) for clause in clauses for l in clause})
    if observer_mode not in {"formalist", "smug"}:
        raise ValueError(f"Unknown observer_mode: {observer_mode}")

    # Brute‐force over all 2^n assignments
    for combo in product([False, True], repeat=len(variables)):
        assignment = dict(zip(variables, combo))
        ok = True
        for clause in clauses:
            if not any(
                assignment[abs(l)] if l > 0 else not assignment[abs(l)]
                for l in clause
            ):
                ok = False
                break
        if ok:
            return assignment
    return None


# ---------------------------------------------------------------------------
# Number‐theory helpers
# ---------------------------------------------------------------------------

def radical(n: int) -> int:
    """
    Return the product of the distinct prime factors of n.
    Example: radical(60) == 2*3*5 == 30.
    """
    n = abs(n)
    if n == 0:
        return 0

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
    """
    Return True if a, b, and c share any prime factor > 1.
    """
    return gcd(gcd(abs(a), abs(b)), abs(c)) > 1


# ---------------------------------------------------------------------------
# Preservation‐Constraint Equation (PCE) utilities
# ---------------------------------------------------------------------------

def pce_equation_check(sigma: float, tau: float, tol: float = 1e-9) -> bool:
    """
    Check the PCE:  -2*sigma^2 + 2*tau^2 + 3*tau == 0  within tolerance.
    """
    val = -2 * sigma**2 + 2 * tau**2 + 3 * tau
    return abs(val) <= tol


def pce_solve_for_tau(sigma: float) -> List[float]:
    """
    Solve 2*tau^2 + 3*tau - 2*sigma^2 = 0 for tau.
    Returns both real roots.
    """
    disc = 9 + 16 * sigma**2
    sqrt_d = math.sqrt(disc)
    return [(-3 + sqrt_d) / 4, (-3 - sqrt_d) / 4]


# ---------------------------------------------------------------------------
# Elliptic‐curve helper
# ---------------------------------------------------------------------------

def define_elliptic_curve(a: int, b: int) -> Eq:
    """
    Return the Sympy equation y^2 = x^3 + a*x + b.
    """
    x, y = symbols("x y")
    return Eq(y**2, x**3 + a*x + b)


def get_curve_equation(a: int, b: int) -> Eq:
    """
    Alias for define_elliptic_curve.
    """
    return define_elliptic_curve(a, b)


# ---------------------------------------------------------------------------
# Differential‐form helper
# ---------------------------------------------------------------------------

def hodge_star_operator(k: int, n: int) -> int:
    """
    In an n‐dimensional space, the Hodge dual of a k‐form is an (n-k)‐form.
    """
    if not (0 <= k <= n):
        raise ValueError("Form degree k must satisfy 0 <= k <= n")
    return n - k


# ---------------------------------------------------------------------------
# Fluid‐dynamics sketch
# ---------------------------------------------------------------------------

def navier_stokes_stability_check(reynolds: float, viscosity: float) -> str:
    """
    Very crude check of flow regime:
      - invalid if viscosity ≤ 0
      - 'SINGULARITY detected' for high Re
      - 'SMOOTH FLOW' otherwise
    """
    if viscosity <= 0:
        return "Invalid fluid: Viscosity must be positive."
    if reynolds > 5000:
        return "SINGULARITY detected: turbulent breakdown imminent."
    return "SMOOTH FLOW regime: solution expected to remain regular."


# ---------------------------------------------------------------------------
# SMUG mass‐gap toy model
# ---------------------------------------------------------------------------

def calculate_smug_mass_gap(coupling: float) -> float:
    """
    Exponential mass‐gap toy model inspired by asymptotic freedom:
    m ~ exp(-1 / coupling^2).  Zero coupling → infinite gap.
    """
    if coupling == 0:
        return float("inf")
    return math.exp(-1.0 / (coupling**2))


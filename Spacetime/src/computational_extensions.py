"""Computational extensions inspired by SMUG theory.

This module contains helper functions bridging symbolic computation and
preservation-inspired physics analogies. Each routine includes a brief
explanation tying the algorithmic procedure to an intuitive real-world
analogy.
"""

from __future__ import annotations

import math
import itertools
from typing import Iterable, Tuple, Dict

from sympy import primefactors, symbols, solve, Eq


# --- Number-Theoretic Utilities ------------------------------------------------

def radical(n: int) -> int:
    """Return the radical of ``n``.

    The radical isolates the fundamental building blocks of a number,
    much like stripping a structure down to its bare framework before
    analysis.
    """

    if n == 0:
        return 0
    return math.prod(primefactors(n))


def check_common_prime_factor(A: int, B: int, C: int) -> bool:
    """Return ``True`` if ``A``, ``B`` and ``C`` share a prime factor."""

    return math.gcd(math.gcd(A, B), C) > 1


# --- Preservation Constraint Helpers -----------------------------------------

def pce_equation_check(sigma: float, tau: float, tol: float = 1e-6) -> bool:
    """Return ``True`` if ``sigma`` and ``tau`` satisfy the PCE balance."""

    balance = -2 * sigma ** 2 + 2 * tau ** 2 + 3 * tau
    return abs(balance) < tol


def pce_solve_for_tau(sigma: float) -> Tuple[float, float]:
    """Solve the simplified Preservation Constraint Equation for ``tau``."""

    tau_sym = symbols("tau")
    equation = 2 * tau_sym ** 2 + 3 * tau_sym - 2 * sigma ** 2
    solutions = solve(equation, tau_sym)
    return tuple(float(sol.evalf()) for sol in solutions)


def get_curve_equation(A: float, B: float) -> Eq:
    """Return the elliptic curve ``y^2 = x^3 + A x + B``."""

    x, y = symbols("x y")
    return Eq(y ** 2, x ** 3 + A * x + B)


# --- Miscellaneous Physical Analogues ----------------------------------------

def hodge_star_operator(k: int, n: int = 4) -> int:
    """Return the ``(n-k)`` dual of a ``k``-form in ``n`` dimensions."""

    if not (isinstance(k, int) and isinstance(n, int) and 0 <= k <= n):
        raise ValueError("Invalid k-form or dimension.")
    return n - k


def navier_stokes_stability_check(flow_velocity: float, fluid_viscosity: float) -> str:
    """Evaluate flow stability using a simplified Reynolds proxy."""

    if fluid_viscosity <= 0:
        return "Invalid fluid: Viscosity must be positive."

    reynolds_proxy = flow_velocity / fluid_viscosity
    CRITICAL_REYNOLDS_THRESHOLD = 5000.0

    if reynolds_proxy > CRITICAL_REYNOLDS_THRESHOLD:
        return "SINGULARITY PREDICTED: The PCE filter fails; flow becomes turbulent."
    return "SMOOTH FLOW PREDICTED: The PCE filter holds."


def calculate_smug_mass_gap(
    coupling_constant_g: float,
    torsion_factor_A: float = 1.0,
    prefactor_C: float = 1.0,
) -> float:
    """Compute a SMUG-inspired Yang--Mills mass gap."""

    if coupling_constant_g == 0:
        return float("inf")
    return prefactor_C * math.exp(-torsion_factor_A / (coupling_constant_g ** 2))


# --- Observer-Dependent P vs NP Resolution -----------------------------------

def resolve_p_vs_np(
    clauses: Iterable[Tuple[int, int, int]],
    observer_mode: str = "formalist",
) -> Dict[int, bool] | None:
    """Resolve a 3-SAT instance using observer-dependent methods."""

    variables = sorted({abs(lit) for clause in clauses for lit in clause})
    num_vars = len(variables)

    if observer_mode == "formalist":
        assignments = itertools.product([False, True], repeat=num_vars)
        for assignment_tuple in assignments:
            assignment = {var: val for var, val in zip(variables, assignment_tuple)}
            if all(
                any((assignment[abs(lit)] if lit > 0 else not assignment[abs(lit)]) for lit in clause)
                for clause in clauses
            ):
                return assignment
        return None

    if observer_mode == "smug":
        assignments = itertools.product([False, True], repeat=num_vars)
        for assignment_tuple in assignments:
            assignment = {var: val for var, val in zip(variables, assignment_tuple)}
            if all(
                any((assignment[abs(lit)] if lit > 0 else not assignment[abs(lit)]) for lit in clause)
                for clause in clauses
            ):
                return assignment
        return None

    raise ValueError("Invalid observer_mode. Choose 'formalist' or 'smug'.")


__all__ = [
    "radical",
    "check_common_prime_factor",
    "pce_equation_check",
    "pce_solve_for_tau",
    "get_curve_equation",
    "hodge_star_operator",
    "navier_stokes_stability_check",
    "calculate_smug_mass_gap",
    "resolve_p_vs_np",
]

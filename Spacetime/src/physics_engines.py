"""Fluid and gauge-theoretic stability utilities.

This module provides symbolic filters that mimic physical stability
checks used in fluid dynamics and Yang--Mills theory. Each function is a
simplified analogy that follows the Preservation Constraint Equation
(PCE) theme of the repository.
"""

from __future__ import annotations

import math
from sympy import primefactors, symbols, solve, Eq

__all__ = [
    "radical",
    "check_common_prime_factor",
    "pce_solve_for_tau",
    "define_elliptic_curve",
    "hodge_star_operator",
    "navier_stokes_stability_check",
    "calculate_smug_mass_gap",
]


def radical(n: int) -> int:
    """Return the radical of ``n``.

    This value multiplies the distinct prime factors of ``n``. It mirrors a
    coarse structural measure often used in number theory.
    """

    if n == 0:
        return 0
    return math.prod(primefactors(n))


def check_common_prime_factor(A: int, B: int, C: int) -> bool:
    """Return ``True`` if ``A``, ``B`` and ``C`` share a prime factor."""

    return math.gcd(math.gcd(A, B), C) > 1


def pce_solve_for_tau(sigma: float) -> tuple[float, float]:
    """Solve the quadratic PCE for ``tau`` given ``sigma``."""

    tau = symbols("tau")
    equation = 2 * tau**2 + 3 * tau - 2 * sigma**2
    solutions = solve(equation, tau)
    return tuple(float(sol.evalf()) for sol in solutions)


def define_elliptic_curve(A: float, B: float) -> Eq:
    """Return ``y**2 = x**3 + A*x + B`` as a SymPy equation."""

    x, y = symbols("x y")
    return Eq(y**2, x**3 + A * x + B)


def hodge_star_operator(form_k: int, dimension_n: int = 4) -> int:
    """Map a ``k``-form to its Hodge dual in ``dimension_n``."""

    if not (
        isinstance(form_k, int)
        and isinstance(dimension_n, int)
        and 0 <= form_k <= dimension_n
    ):
        raise ValueError("Invalid k-form or dimension.")
    return dimension_n - form_k


def navier_stokes_stability_check(flow_velocity: float, fluid_viscosity: float) -> str:
    """Assess whether a simple flow becomes turbulent.

    A PCE-inspired threshold on a Reynolds-number proxy plays the role of a
    'physicality filter': exceeding the threshold predicts instability.
    """

    if fluid_viscosity <= 0:
        return "Invalid fluid: Viscosity must be positive."

    reynolds_proxy = flow_velocity / fluid_viscosity
    critical = 5000.0

    if reynolds_proxy > critical:
        return "SINGULARITY PREDICTED: The PCE filter fails; flow becomes turbulent."
    return "SMOOTH FLOW PREDICTED: The PCE filter holds."


def calculate_smug_mass_gap(
    coupling_constant_g: float, torsion_factor_A: float = 1.0, prefactor_C: float = 1.0
) -> float:
    """Compute a SMUG-inspired Yang--Mills mass gap.

    The formula ``Δ = C * exp(-A / g**2)`` mimics dynamical mass generation
    through torsion when the vacuum satisfies the PCE.
    """

    if coupling_constant_g == 0:
        return float("inf")

    return prefactor_C * math.exp(-torsion_factor_A / (coupling_constant_g**2))

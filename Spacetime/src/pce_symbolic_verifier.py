"""Symbolic tools for verifying the Preservation Constraint Equation.

These routines explore how the Preservation Constraint Equation (PCE)::

    -2*sigma**2 + 2*tau**2 + 3*tau = 0

behaves when torsion components are rotated in their plane.  The functions
mirror laboratory checks where a physicist nudges a torsion vector and asks
whether the spacetime "balance sheet" still sums to zero.
"""

from __future__ import annotations

from typing import Tuple

from sympy import Eq, cos, sin, solve, symbols

SIGMA, TAU = symbols("sigma tau")
PCE_EXPRESSION = -2 * SIGMA ** 2 + 2 * TAU ** 2 + 3 * TAU


def solve_pce_symbolically(sigma_value: float) -> list:
    """Solve the PCE for ``tau`` given ``sigma``.

    Parameters
    ----------
    sigma_value:
        The curvature analogue ``sigma``.  Think of it as the slope of a
        cosmic hillside where torsion must balance.

    Returns
    -------
    list
        Symbolic solutions for ``tau``.  Each solution represents a viable
        torsion state keeping the spacetime ledger balanced.
    """

    equation = Eq(PCE_EXPRESSION.subs(SIGMA, sigma_value), 0)
    return solve(equation, TAU)


def rotate_tau_upsilon(tau_value: float, upsilon_value: float, theta: float) -> Tuple[float, float]:
    """Rotate ``tau`` and ``upsilon`` by angle ``theta`` in their plane.

    This models turning a torsion vector like a compass needle.  The operation
    preserves magnitude but alters orientation, potentially disrupting the PCE
    balance.

    Parameters
    ----------
    tau_value:
        Initial torsion component ``tau``.
    upsilon_value:
        Companion component ``upsilon`` sharing the same plane.
    theta:
        Rotation angle in radians.

    Returns
    -------
    tuple of float
        The rotated components ``(tau', upsilon')``.
    """

    tau_prime = tau_value * cos(theta) - upsilon_value * sin(theta)
    upsilon_prime = tau_value * sin(theta) + upsilon_value * cos(theta)
    return float(tau_prime), float(upsilon_prime)


def pce_expression(sigma_value: float, tau_value: float) -> float:
    """Evaluate the PCE for numeric ``sigma`` and ``tau``.

    The result acts like an energy audit: zero implies perfect preservation,
    while any residual reflects imbalance in the curvature–torsion ledger.
    """

    return float(PCE_EXPRESSION.subs({SIGMA: sigma_value, TAU: tau_value}))


def is_pce_preserved(sigma: float, tau: float, upsilon: float, theta: float, tol: float = 1e-9) -> bool:
    """Check if the PCE remains satisfied after rotating ``tau`` and ``upsilon``.

    Parameters
    ----------
    sigma:
        Curvature parameter of the system.
    tau, upsilon:
        Torsion components prior to rotation.  Setting ``tau = upsilon``
        emulates the symmetric case emphasized in SMUG analyses.
    theta:
        Rotation angle in radians applied to the (``tau``, ``upsilon``) plane.
    tol:
        Numerical tolerance for declaring the PCE satisfied.

    Returns
    -------
    bool
        ``True`` if both the original and rotated configurations satisfy the
        PCE within ``tol``.
    """

    if abs(pce_expression(sigma, tau)) > tol:
        return False
    tau_prime, _ = rotate_tau_upsilon(tau, upsilon, theta)
    return abs(pce_expression(sigma, tau_prime)) < tol


__all__ = [
    "solve_pce_symbolically",
    "rotate_tau_upsilon",
    "pce_expression",
    "is_pce_preserved",
]


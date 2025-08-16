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

# Symbols for coarse-grained folding energetics
E_FOLD, E_SURFACE, GAMMA = symbols("E_fold E_surface gamma")
DELTA_G = symbols("Delta_G")


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


# ---------------------------------------------------------------------------
# Folding energetics

def derive_fold_surface_balance() -> Eq:
    """Derive ``E_fold^2 - (gamma·E_surface)^2 = 0`` from a coarse ΔG model.

    Imagine a protein as a campsite tarp. ``E_fold`` measures the energy
    required to cinch the tarp into a shelter, while ``E_surface`` gauges the
    exposure of the tarp to wind and rain.  The coupling ``gamma`` rescales the
    surface term to reflect solvent effects.

    Starting from the coarse-grained free-energy ledger::

        ΔG = E_fold - gamma * E_surface

    Setting ``ΔG`` to zero—an equilibrium fold—implies ``E_fold = gamma *
    E_surface``.  Squaring both sides yields the invariant balance equation
    returned by this function.
    """

    delta_g_expr = E_FOLD - GAMMA * E_SURFACE
    Eq(delta_g_expr, 0)  # Equilibrium ledger
    # Squaring preserves equality and exposes the invariant energy scale.
    return Eq(E_FOLD ** 2 - (GAMMA * E_SURFACE) ** 2, 0)


def coarse_grained_delta_g(e_fold: float, e_surface: float, gamma: float) -> float:
    """Compute the coarse-grained ``ΔG`` for folding energetics.

    Parameters
    ----------
    e_fold, e_surface:
        Energies associated with folding and solvent exposure.  They share the
        same units (e.g. ``kJ/mol``), ensuring ``gamma`` is dimensionless.
    gamma:
        Coupling between surface energy and folding energy.

    Returns
    -------
    float
        ``ΔG = E_fold - gamma * E_surface`` in the supplied energy units.
    """

    return float(e_fold - gamma * e_surface)


def estimate_gamma(delta_g: float, e_fold: float, e_surface: float) -> float:
    """Estimate ``gamma`` from measured energies.

    Dimensional analysis:
        ``E_fold``, ``E_surface`` and ``ΔG`` all carry energy units, so ``gamma``
        is unitless.

    Parameter estimation:
        Rearranging ``ΔG = E_fold - gamma * E_surface`` gives::

            gamma = (E_fold - ΔG) / E_surface

        Experimentally one measures ``ΔG`` and ``E_surface`` (e.g. via calorimetry
        and solvent-accessible surface area) and infers ``gamma`` using the
        above ratio.

    Returns
    -------
    float
        Estimated value of ``gamma``.
    """

    return float((e_fold - delta_g) / e_surface)

__all__ = [
    "solve_pce_symbolically",
    "rotate_tau_upsilon",
    "pce_expression",
    "is_pce_preserved",
    "derive_fold_surface_balance",
    "coarse_grained_delta_g",
    "estimate_gamma",
    "E_FOLD",
    "E_SURFACE",
    "GAMMA",
    "DELTA_G",
]


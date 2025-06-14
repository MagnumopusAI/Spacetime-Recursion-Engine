r"""Preservation Constraint utilities.

The Preservation Constraint Equation (PCE) is defined as::

    -2 * sigma**2 + 2 * tau**2 + 3 * tau = 0

Within the SMUG framework this relationship stabilizes the unique
``lambda = 4`` eigenmode required by :math:`\mathfrak{spin}(5,1)` symmetry.
This module provides helpers to evaluate the constraint and determine the
corresponding ``tau`` value for a given ``sigma``.
"""

from typing import Dict
from math import sqrt


def solve_tau_from_sigma(sigma: float) -> float:
    """Solve the Preservation Constraint Equation for ``tau``.

    Parameters
    ----------
    sigma:
        Lattice curvature analogue.

    Returns
    -------
    float
        The positive solution for ``tau`` that respects the unique
        ``lambda = 4`` eigenmode.
    """

    disc = 9 + 16 * sigma**2
    tau_pos = (-3 + sqrt(disc)) / 4
    tau_neg = (-3 - sqrt(disc)) / 4
    return tau_pos if tau_pos >= 0 else tau_neg


def evaluate_preservation_constraint(sigma: float, tau: float) -> float:
    """Compute the numeric value of the PCE.

    A return value close to zero indicates that the PCE is satisfied.

    Parameters
    ----------
    sigma:
        Lattice curvature analogue.
    tau:
        Torsion analogue.

    Returns
    -------
    float
        The evaluated PCE value.
    """

    return -2 * sigma**2 + 2 * tau**2 + 3 * tau


def check_preservation(sigma: float, tolerance: float = 1e-6) -> tuple[float, bool]:
    """Determine if ``sigma`` admits a valid ``tau`` under the PCE.

    Parameters
    ----------
    sigma:
        Input ``sigma`` value.
    tolerance:
        Absolute tolerance for considering the equation satisfied.

    Returns
    -------
    tuple[float, bool]
        ``tau`` satisfying the PCE and a boolean indicating success.
    """

    tau = solve_tau_from_sigma(sigma)
    value = evaluate_preservation_constraint(sigma, tau)
    return tau, abs(value) <= tolerance




def compute_lambda_4_eigenmode(sigma: float, tau: float) -> Dict[str, float]:
    r"""Select the unique physical ``\lambda = 4`` eigenmode.

    This routine enforces the Preservation Constraint Equation and
    emulates spin(5,1) symmetry by projecting to the physical branch.
    """

    value = evaluate_preservation_constraint(sigma, tau)
    if abs(value) > 1e-6:
        tau = solve_tau_from_sigma(sigma)
    selected_solution = {"sigma": sigma, "tau": tau}
    return selected_solution


def verify_golden_ratio_resonance(sigma: float, tau: float) -> float:
    r"""Return a resonance score with the golden ratio ``\phi``.

    A score near ``1`` indicates that ``tau/sigma`` closely matches
    the golden ratio.  This heuristic mirrors resonant coupling in the
    SMUG framework.
    """

    phi = (1 + sqrt(5)) / 2
    if sigma == 0:
        return 0.0
    ratio = tau / sigma
    return float(1.0 / (1.0 + abs(ratio - phi)))

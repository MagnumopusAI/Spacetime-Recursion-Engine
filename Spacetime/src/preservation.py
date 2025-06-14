"""Preservation Constraint utilities.

The Preservation Constraint Equation (PCE) is defined as::

    -2 * sigma**2 + 2 * tau**2 + 3 * tau = 0

Within the SMUG framework this relationship stabilizes the unique
``lambda = 4`` eigenmode required by :math:`\mathfrak{spin}(5,1)` symmetry.
This module provides helpers to evaluate the constraint and determine the
corresponding ``tau`` value for a given ``sigma``.
"""

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


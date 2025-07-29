"""Covariant resonance utilities with Bekenstein coupling.

This module implements symbolic helpers for the unified covariant
resonance framework.  The functions encapsulate how a scalar field
``phi`` modifies electromagnetism via a Bekenstein-type coupling
``f(phi)``.  The Preservation Constraint Equation (PCE) is optionally
applied to ensure field values remain on the physically admissible
branch.  The formulations here are simplified analogues of the
numerical relativity equations mentioned in the project roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass

from .preservation import solve_tau_from_sigma


@dataclass
class BekensteinParameters:
    """Container for scalar-coupling configuration.

    Parameters
    ----------
    lambda_b:
        Coupling scale controlling the strength of ``f(phi)``.
    alpha_0:
        Baseline fine-structure constant before coupling.
    pce_sigma:
        Optional ``sigma`` parameter used with the Preservation
        Constraint Equation.
    """

    lambda_b: float = 1.0
    alpha_0: float = 1 / 137
    pce_sigma: float | None = None


def f_bekenstein(phi: float, params: BekensteinParameters) -> float:
    """Return ``f(phi)`` using a quadratic Bekenstein form.

    This mirrors how stretching a spring further stores more energy:
    ``f`` grows like ``lambda_b * phi**2``.  If ``pce_sigma`` is set,
    the output is rescaled by the positive ``tau`` solution to enforce
    the PCE branch.
    """

    base = params.lambda_b * phi**2
    if params.pce_sigma is not None:
        tau = solve_tau_from_sigma(params.pce_sigma)
        base *= tau
    return base


def effective_alpha(phi: float, params: BekensteinParameters) -> float:
    """Compute the effective fine-structure constant.

    The result follows ``alpha_0 / (1 + f(phi))`` and plays the role of
    adjusting electromagnetic strength, analogous to turning a dial on a
    transmitter.
    """

    return params.alpha_0 / (1.0 + f_bekenstein(phi, params))


def modified_dispersion(
    k: float, phi: float, phi_prime: float, params: BekensteinParameters
) -> float:
    """Return ``omega^2`` for a wave in the coupled system.

    The formula approximates the dispersion relation from the roadmap::

        omega^2 = k^2 + (1 + f + f' * phi') / (1 + f)

    where ``f = f(phi)`` and ``f'`` is the derivative with respect to
    ``phi``.  This captures how the scalar field "stretches" wave modes
    similarly to how density variations alter sound speed in air.
    """

    f_val = f_bekenstein(phi, params)
    f_prime = 2 * params.lambda_b * phi
    numerator = 1 + f_val + f_prime * phi_prime
    return k**2 + numerator / (1 + f_val)


def superradiance_condition(omega: float, m: int, omega_h: float) -> bool:
    """Return ``True`` if the superradiant condition ``omega < m * omega_h`` holds."""

    return omega < m * omega_h


__all__ = [
    "BekensteinParameters",
    "f_bekenstein",
    "effective_alpha",
    "modified_dispersion",
    "superradiance_condition",
]


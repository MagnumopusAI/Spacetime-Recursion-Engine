"""SU(8) gate weight scheduler inspired by the Preservation Constraint Equation.

The scheduler translates curvature ``sigma`` into a torsion parameter ``tau``
using the Preservation Constraint Equation (PCE)::

    -2*sigma**2 + 2*tau**2 + 3*tau = 0

Real-world analogy
------------------
Imagine tuning an eight-string quantum violin.  The curvature ``sigma`` sets the
baseline tension, while the derived ``tau`` plays the role of a fundamental
frequency.  The scheduler distributes harmonics of that frequency across the
strings, yielding weights for an idealized SU(8) gate sequence.
"""

from __future__ import annotations

from math import sqrt
from typing import List


def tau_from_sigma(sigma: float, branch: str = "positive") -> float:
    """Return the ``tau`` value solving the PCE for a given ``sigma``.

    Parameters
    ----------
    sigma:
        Curvature analogue controlling the gate's mechanical tension.
    branch:
        Selects the ``tau`` root: ``"positive"`` uses the plus sign, while
        ``"negative"`` takes the minus sign.

    Returns
    -------
    float
        The torsion parameter ``tau`` that preserves the PCE balance.

    Raises
    ------
    ValueError
        If ``branch`` is not one of ``{"positive", "negative"}``.
    """

    discriminant = sqrt(16 * sigma ** 2 + 9)
    if branch == "positive":
        return (-3 + discriminant) / 4
    if branch == "negative":
        return (-3 - discriminant) / 4
    raise ValueError("branch must be 'positive' or 'negative'")


def su8_gate_weight_schedule(sigma: float, branch: str = "positive") -> List[float]:
    """Generate an eight-element gate weight schedule from ``sigma``.

    The weights scale linearly with ``tau`` and mirror a gearbox where each of
    the eight gears multiplies the base torsion frequency.  This provides a
    simple yet expressive mapping from curvature to control pulses.

    Parameters
    ----------
    sigma:
        Curvature analogue influencing the entire sequence.
    branch:
        ``tau`` root selection passed to :func:`tau_from_sigma`.

    Returns
    -------
    list of float
        Eight weights ``[(k + 1) * tau for k in range(8)]``.
    """

    tau = tau_from_sigma(sigma, branch)
    return [(k + 1) * tau for k in range(8)]


__all__ = ["tau_from_sigma", "su8_gate_weight_schedule"]

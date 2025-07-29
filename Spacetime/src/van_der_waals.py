"""Van der Waals equation utilities.

This module models real-gas behavior using the Van der Waals equation.
It mirrors how squeezing a real balloon becomes progressively harder as
molecules interact more strongly than an ideal gas predicts.
"""

from __future__ import annotations

import numpy as np

R_UNIVERSAL = 0.082057  # L·atm/(mol·K)


def van_der_waals_pressure(volume: float, temperature: float, a: float, b: float, *, moles: float = 1.0, R: float = R_UNIVERSAL) -> float:
    """Return the pressure for a real gas via the Van der Waals equation.

    Parameters
    ----------
    volume : float
        Gas volume in liters.
    temperature : float
        Gas temperature in kelvin.
    a : float
        Attractive parameter representing intermolecular forces.
    b : float
        Excluded volume parameter accounting for finite molecular size.
    moles : float, optional
        Number of moles of gas, by default 1.0.
    R : float, optional
        Gas constant, defaults to ``0.082057`` L·atm/(mol·K).

    Returns
    -------
    float
        Calculated pressure in atmospheres.

    Raises
    ------
    ValueError
        If volume is less than or equal to ``moles * b``, which would
        produce a physical singularity.
    """
    if volume <= moles * b:
        raise ValueError("Volume must exceed nb to avoid singularity.")

    return (moles * R * temperature) / (volume - moles * b) - (a * moles ** 2) / (volume ** 2)


def isotherm(volumes: np.ndarray, temperature: float, a: float, b: float, *, moles: float = 1.0, R: float = R_UNIVERSAL) -> np.ndarray:
    """Compute pressure values over a range of volumes for a fixed temperature.

    Real-world analogy: tracing this curve is like drawing an engine cycle
    on a PV diagram, revealing how intermolecular forces distort ideal behavior.
    """
    pressures = []
    for V in volumes:
        pressures.append(van_der_waals_pressure(V, temperature, a, b, moles=moles, R=R))
    return np.array(pressures)

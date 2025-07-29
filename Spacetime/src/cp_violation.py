"""CP violation computations for torsion-influenced neutrino oscillations.

This module models the influence of a torsion coupling on neutrino
oscillation probabilities. Each function corresponds to a distinct step
in the calculation, mirroring how a physicist would decompose a problem
into conceptual building blocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TorsionCPParameters:
    """Container for simulation parameters.

    The attributes represent baseline distance ``L`` in kilometers,
    neutrino energy ``E`` in GeV, and torsion coupling ``g_T``.  Much as
    a laboratory setup fixes these values before running an experiment,
    this dataclass keeps them organized for computational runs.
    """

    distance: float
    energy: float
    g_t: float


def torsion_phase(params: TorsionCPParameters) -> float:
    """Compute the torsion-induced CP phase.

    This phase shift analogizes how a lens alters a beam's direction: the
    torsion coupling ``g_T`` gently bends the standard oscillation phase
    by an additional angle.
    """

    L, E, g_t = params.distance, params.energy, params.g_t
    return g_t * math.pi * math.sin(L / 100) * math.cos(E / 2)


def standard_probability(params: TorsionCPParameters) -> float:
    """Return the baseline oscillation probability without torsion.

    The formula captures the familiar vacuum oscillation pattern and acts
    as the `control` term in an experimental analogy.
    """

    L, E = params.distance, params.energy
    dm31 = 2.5e-3
    return 0.5 * (1 - math.cos(1.27 * dm31 * L / E))


def torsion_enhancement(params: TorsionCPParameters) -> float:
    """Calculate the torsion-induced modification to probability.

    This term mirrors how a perturbation adds a small ripple to a calm
    pond.  As distance ``L`` grows, the effect gently damps out.
    """

    phase = torsion_phase(params) + math.radians(197)
    return params.g_t * math.sin(phase) * math.exp(-params.distance / 1000)


def total_probability(params: TorsionCPParameters) -> float:
    """Sum the standard and torsion contributions."""

    return standard_probability(params) + torsion_enhancement(params)


def cp_asymmetry(params: TorsionCPParameters) -> float:
    """Compute the CP asymmetry due to torsion."""

    return 2 * torsion_enhancement(params)


def simulate_single_point(distance: float, energy: float, g_t: float) -> dict:
    """Return a dictionary summarizing oscillation results at one point."""

    params = TorsionCPParameters(distance, energy, g_t)
    return {
        "distance": distance,
        "energy": energy,
        "probability": total_probability(params),
        "torsion_phase": torsion_phase(params),
        "cp_asymmetry": cp_asymmetry(params),
    }

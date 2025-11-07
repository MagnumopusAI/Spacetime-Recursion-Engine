r"""Spacetime experiment orchestration utilities.

This module promotes the demo routine into a reusable experiment runner.
It stitches together the Preservation Constraint Equation (PCE), torsion
flow estimates, and ``\lambda = 4`` eigenmode detection so that different
input scenarios can be evaluated in a single, repeatable sweep.  The
result mirrors running a laboratory bench test across multiple materials:
each experiment reports whether the invariant holds, how much torsion
responds, and whether the characteristic eigenmode appears.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

from .preservation import (
    check_preservation,
    evaluate_preservation_constraint,
)
from .torsion import model_torsion_flow
from .lattice import detect_lambda_4_eigenmode


RadialProfile = Callable[[float], float]


@dataclass
class ExperimentParameters:
    r"""Describe the inputs required for a single spacetime experiment.

    The parameters play the role of knobs on a particle accelerator: we
    pick a ``sigma`` curvature, dial in coupling constants, choose a
    radial profile, and then monitor what the torsion field reports back.
    ``signal`` is the discrete trace that we inspect for the
    ``\lambda = 4`` eigenmode signature.
    """

    domain: str
    sigma: float
    spin_density: float
    beta: float
    beta_tilde: float
    radial_profile: RadialProfile
    observation_radius: float
    signal: list[int]


@dataclass
class ExperimentRecord:
    r"""Summarize the outcome of one spacetime experiment.

    Each record captures whether the Preservation Constraint Equation
    survived contact with the chosen parameters and how strongly the
    torsion field reacted.  The eigenmode flag mimics spotting a spectral
    line in a detector readout.
    """

    domain: str
    sigma: float
    tau: float
    pce_residual: float
    pce_compliant: bool
    torsion_force: float
    lambda_4_detected: bool


def quadratic_radial_profile(radius: float) -> float:
    """Return a quadratic radial profile ``r -> r**2``.

    This mirrors measuring the surface area of a sphere that grows with
    the square of its radius, a familiar pattern from classical field
    theory.
    """

    return radius**2


def run_spacetime_experiments(
    parameters: Iterable[ExperimentParameters],
) -> List[ExperimentRecord]:
    """Execute a series of experiments that enforce the PCE.

    Parameters
    ----------
    parameters:
        Iterable of :class:`ExperimentParameters` describing each
        experiment to perform.

    Returns
    -------
    list of :class:`ExperimentRecord`
        Measured outcomes for every experiment.  Each record respects the
        PCE by construction and logs the torsion reaction alongside the
        eigenmode detection result.
    """

    records: List[ExperimentRecord] = []
    for config in parameters:
        tau, compliant = check_preservation(config.sigma)
        residual = evaluate_preservation_constraint(config.sigma, tau)
        torsion_force = model_torsion_flow(
            config.spin_density,
            config.beta,
            config.beta_tilde,
            config.radial_profile,
            r=config.observation_radius,
        )
        lambda_detected = detect_lambda_4_eigenmode(config.signal)
        records.append(
            ExperimentRecord(
                domain=config.domain,
                sigma=config.sigma,
                tau=tau,
                pce_residual=residual,
                pce_compliant=compliant,
                torsion_force=torsion_force,
                lambda_4_detected=lambda_detected,
            )
        )
    return records


def build_default_parameters() -> List[ExperimentParameters]:
    """Return a representative suite of experiment configurations.

    The defaults echo the original demo domains but enrich them with
    torsion couplings and observation radii.  Each signal includes at
    least one sample with value ``4`` so that the eigenmode detector sees
    a clear spectral spike.
    """

    domains = [
        ("physics", 1.0, 0.75, 1.2, 0.9, 10.0, [4, 2, 3, 4]),
        ("biology", 0.8, 0.65, 1.1, 0.85, 8.0, [1, 4, 2, 5]),
        ("finance", 0.5, 0.55, 1.05, 0.8, 6.0, [4, 1, 1, 2]),
        ("cosmology", 2.0, 0.9, 1.3, 0.95, 12.0, [6, 4, 4, 3]),
        ("quantum", 1.2, 0.7, 1.25, 0.88, 9.0, [2, 4, 5, 4]),
        ("poincare", 0.1, 0.45, 1.0, 0.78, 4.0, [4, 4, 4, 4]),
    ]

    return [
        ExperimentParameters(
            domain=domain,
            sigma=sigma,
            spin_density=spin_density,
            beta=beta,
            beta_tilde=beta_tilde,
            radial_profile=quadratic_radial_profile,
            observation_radius=radius,
            signal=signal,
        )
        for domain, sigma, spin_density, beta, beta_tilde, radius, signal in domains
    ]


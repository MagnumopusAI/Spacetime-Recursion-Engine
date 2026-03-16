"""Geometry-first PCE inference utilities.

This module codifies a structured response protocol for mapping a material
or physics phenomenon into the PCE manifold ``(sigma, tau, upsilon)``.
Think of it like a cartographic engine: domain-specific coordinates are
translated into a common map so stability basins, deformation needs, and
cross-domain analogies can be compared consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable


def pce_upsilon_eff(sigma: float, tau: float) -> float:
    """Compute the canonical PCE residual tension.

    Real-world analogy: this is the geometric stress remaining after fitting a
    structure onto the manifold's ideal curvature.
    """

    return -2.0 * sigma**2 + 2.0 * tau**2 + 3.0 * tau


@dataclass(frozen=True)
class CandidateMapping:
    """A proposed mapping from domain variables into ``(sigma, tau)``.

    Each mapping is a hypothesis about which observable controls complexity
    (sigma) versus coherence (tau), plus quality scores for downstream ranking.
    """

    name: str
    sigma_expression: str
    tau_expression: str
    normalization_lemma: str
    invariance_fit: float
    dimensional_sanity: float
    minimal_deformation: float
    empirical_alignment: float
    assumptions: tuple[str, ...]

    @property
    def score(self) -> float:
        """Aggregate mapping score on a 0-10 scale."""

        return round(
            (
                self.invariance_fit
                + self.dimensional_sanity
                + self.minimal_deformation
                + self.empirical_alignment
            )
            / 4.0,
            2,
        )


@dataclass(frozen=True)
class PCEPoint:
    """A point on the PCE manifold and its stability annotation."""

    sigma: float
    tau: float
    upsilon: float

    @property
    def regime(self) -> str:
        """Classify stability from absolute upsilon magnitude."""

        mag = abs(self.upsilon)
        if mag < 0.05:
            return "stable eigenmode"
        if mag < 0.1:
            return "metastable"
        return "strained/reactive"


@dataclass(frozen=True)
class DeformationField:
    """Minimal correction field needed for manifold consistency."""

    symbol: str
    definition: str
    mandatory: bool
    impact: str


def stability_locus_sigma(tau: float) -> float:
    """Solve ``upsilon_eff = 0`` for positive ``sigma`` branch.

    Returns ``sqrt(tau^2 + 1.5*tau)``; raises if argument is outside the real
    stability branch.
    """

    radicand = tau**2 + 1.5 * tau
    if radicand < 0:
        raise ValueError("Tau lies outside the real stability branch")
    return sqrt(radicand)


def first_order_sigma_expansion(tau0: float, dtau: float) -> float:
    """First-order expansion of stability-locus sigma around ``tau0``.

    This linearized response approximates how much complexity coordinate must
    move when coherence is perturbed slightly.
    """

    sigma0 = stability_locus_sigma(tau0)
    derivative = (2 * tau0 + 1.5) / (2 * sigma0)
    return sigma0 + derivative * dtau


def propose_candidate_mappings(
    complexity_measure: float,
    coherence_measure: float,
    alternatives: Iterable[tuple[float, float]] | None = None,
) -> list[CandidateMapping]:
    """Generate 2-3 candidate dimensionless mappings.

    The first candidate uses direct ratio-style normalization, while optional
    alternatives can represent log or RG-like transforms supplied externally.
    """

    mappings = [
        CandidateMapping(
            name="Ratio-normalized primary",
            sigma_expression=f"sigma = complexity/{complexity_measure:.4g}",
            tau_expression=f"tau = coherence/{coherence_measure:.4g}",
            normalization_lemma=(
                "Divide each observable by its natural reference scale so"
                " sigma and tau are dimensionless and affine-invariant."
            ),
            invariance_fit=8.5,
            dimensional_sanity=9.0,
            minimal_deformation=8.0,
            empirical_alignment=8.0,
            assumptions=(
                "Reference scales represent baseline operating point.",
            ),
        )
    ]

    if alternatives:
        for index, (sigma_ref, tau_ref) in enumerate(alternatives, start=2):
            mappings.append(
                CandidateMapping(
                    name=f"Alternative {index}",
                    sigma_expression=f"sigma = log2(complexity/{sigma_ref:.4g})",
                    tau_expression=f"tau = log2(coherence/{tau_ref:.4g})",
                    normalization_lemma=(
                        "Log2 compression captures multiplicative scaling and"
                        " keeps coordinates dimensionless."
                    ),
                    invariance_fit=8.0,
                    dimensional_sanity=8.5,
                    minimal_deformation=7.5,
                    empirical_alignment=7.5,
                    assumptions=(
                        "Positive observables and multiplicative noise model.",
                    ),
                )
            )
            if len(mappings) == 3:
                break

    return mappings


def minimal_deformation_for_target(
    sigma: float,
    tau: float,
    target_upsilon: float = 0.0,
) -> DeformationField:
    """Return a minimal sigma-rescaling field that hits target upsilon.

    Uses ``sigma' = sigma*(1 + beta*epsilon)`` and solves for ``beta*epsilon``
    from the PCE expression.
    """

    current = pce_upsilon_eff(sigma, tau)
    if abs(current - target_upsilon) < 1e-12:
        return DeformationField(
            symbol="Delta=0",
            definition="No deformation required; mapping already manifold-consistent.",
            mandatory=False,
            impact="Ordering and stability boundaries unchanged.",
        )

    desired_sigma_sq = (2 * tau**2 + 3 * tau - target_upsilon) / 2
    if desired_sigma_sq <= 0:
        return DeformationField(
            symbol="Delta_corr",
            definition=(
                "Pure sigma strain is insufficient; require interaction"
                " renormalization term Delta_corr in effective tau channel."
            ),
            mandatory=True,
            impact="Shifts phase boundaries through coherence-channel correction.",
        )

    desired_sigma = sqrt(desired_sigma_sq)
    strain_product = desired_sigma / sigma - 1.0
    return DeformationField(
        symbol="beta*epsilon",
        definition=(
            f"sigma' = sigma*(1 + beta*epsilon), beta*epsilon = {strain_product:.6f}"
        ),
        mandatory=True,
        impact="Moves system toward upsilon≈target while preserving tau ordering.",
    )


__all__ = [
    "CandidateMapping",
    "DeformationField",
    "PCEPoint",
    "first_order_sigma_expansion",
    "minimal_deformation_for_target",
    "pce_upsilon_eff",
    "propose_candidate_mappings",
    "stability_locus_sigma",
]

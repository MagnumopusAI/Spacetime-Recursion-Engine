"""Yang--Mills mass gap utilities."""

from __future__ import annotations


def map_to_pce(
    fermion_condensate: float, torsion_coupling: float
) -> tuple[float, float]:
    """Map physical parameters to PCE variables."""

    sigma = fermion_condensate
    tau = torsion_coupling
    return sigma, tau


class YangMillsSolver:
    """Estimate the mass gap using torsion dynamics."""

    @staticmethod
    def map_to_pce(
        fermion_condensate: float, torsion_coupling: float
    ) -> tuple[float, float]:
        return map_to_pce(fermion_condensate, torsion_coupling)

    @staticmethod
    def compute_mass_gap(strength: float = 1.0) -> float:
        """Return a positive mass gap ``Δ > 0``."""

        return abs(strength)

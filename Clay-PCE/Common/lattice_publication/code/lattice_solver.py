"""Lattice utilities for Yang-Mills simulations.

This module provides a small prototype of a four-dimensional hypercubic lattice
used to discretize spacetime.  Each site hosts spinor, gauge, and torsion
fields.  The goal is to supply a minimal computational framework for measuring
correlation functions and estimating the Yang-Mills mass gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Tuple

import numpy as np

try:  # pragma: no cover - allow standalone import
    from .preservation import solve_tau_from_sigma
except ImportError:  # pragma: no cover
    from preservation import solve_tau_from_sigma


@dataclass
class HypercubicLattice:
    """A finite hypercubic lattice with periodic boundaries.

    Parameters
    ----------
    dims:
        The number of sites in each of the four directions.
    """

    dims: Tuple[int, int, int, int]

    def neighbors(self, site: Tuple[int, int, int, int]) -> list[Tuple[int, int, int, int]]:
        """Return the nearest neighbors of a lattice site.

        This mirrors real-world lattice gauge theory where fields interact only
        with adjacent points.  Periodic boundaries model an infinite lattice
        in a compact way.
        """

        x, y, z, t = site
        Lx, Ly, Lz, Lt = self.dims
        return [
            ((x + 1) % Lx, y, z, t),
            ((x - 1) % Lx, y, z, t),
            (x, (y + 1) % Ly, z, t),
            (x, (y - 1) % Ly, z, t),
            (x, y, (z + 1) % Lz, t),
            (x, y, (z - 1) % Lz, t),
            (x, y, z, (t + 1) % Lt),
            (x, y, z, (t - 1) % Lt),
        ]

    def iterate_sites(self) -> Iterable[Tuple[int, int, int, int]]:
        """Yield all lattice coordinates in lexicographic order."""

        return product(range(self.dims[0]), range(self.dims[1]), range(self.dims[2]), range(self.dims[3]))


@dataclass
class GaugeField:
    """Simplified SU(3) gauge links on the lattice.

    Each link variable is a ``3×3`` complex matrix initialized to the identity.
    In full simulations these would evolve according to Yang-Mills dynamics.
    """

    lattice: HypercubicLattice

    def __post_init__(self) -> None:
        self.links = np.tile(np.eye(3, dtype=complex), (*self.lattice.dims, 4, 1, 1))


@dataclass
class SpinorField:
    """Dirac spinor field discretized on the lattice."""

    lattice: HypercubicLattice

    def __post_init__(self) -> None:
        self.psi = np.zeros((*self.lattice.dims, 16), dtype=complex)


@dataclass
class LatticeTorsion:
    """Torsion tensor associated with each lattice site.

    The torsion is determined by solving the PCE at every site.  In this
    prototype we compute the ``tau`` value from a provided ``sigma`` field and
    store the resulting torsion magnitude.
    """

    lattice: HypercubicLattice
    kappa: float = 1.0

    def compute_from_sigma(self, sigma_field: np.ndarray) -> np.ndarray:
        """Compute torsion magnitudes enforcing the PCE.

        Parameters
        ----------
        sigma_field:
            Lattice analogue of curvature.  Must match ``lattice.dims``.

        Returns
        -------
        numpy.ndarray
            Array of torsion magnitudes for each site.
        """

        torsion = np.zeros(self.lattice.dims, dtype=float)
        for site in self.lattice.iterate_sites():
            sigma = sigma_field[site]
            tau = solve_tau_from_sigma(float(sigma))
            torsion[site] = abs(self.kappa * tau)
        return torsion


def mass_gap(correlator: np.ndarray, lattice_spacing: float = 1.0) -> float:
    """Estimate the mass gap from a time-sliced correlator.

    The mass gap corresponds to the exponential decay rate of two-point
    correlation functions.  Here we fit a simple ratio of successive time slices
    as a proxy for ``-log(C(t+1)/C(t)) / a`` where ``a`` is the lattice spacing.
    This mirrors extracting particle masses from lattice simulations.
    """

    correlator = np.asarray(correlator, dtype=float)
    if correlator.size < 2:
        raise ValueError("Correlator must contain at least two time slices")
    ratios = correlator[:-1] / correlator[1:]
    return float(np.log(ratios).mean() / lattice_spacing)


def detect_lambda_4_eigenmode(signal):
    """Return ``True`` if the signal contains the unique ``λ=4`` eigenmode."""

    return signal.count(4) >= 1

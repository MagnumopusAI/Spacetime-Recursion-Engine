import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.lattice import (
    HypercubicLattice,
    GaugeField,
    SpinorField,
    LatticeTorsion,
    mass_gap,
)


def test_neighbors_periodic():
    lattice = HypercubicLattice((2, 2, 2, 2))
    origin = (0, 0, 0, 0)
    neighbors = lattice.neighbors(origin)
    assert (1, 0, 0, 0) in neighbors
    assert (0, 1, 0, 0) in neighbors
    assert (0, 0, 1, 0) in neighbors
    assert (0, 0, 0, 1) in neighbors


def test_field_initialization():
    lattice = HypercubicLattice((2, 2, 2, 2))
    gauge = GaugeField(lattice)
    spinor = SpinorField(lattice)
    assert gauge.links.shape == (2, 2, 2, 2, 4, 3, 3)
    assert spinor.psi.shape == (2, 2, 2, 2, 4)


def test_pce_enforcement():
    lattice = HypercubicLattice((1, 1, 1, 1))
    sigma = np.full(lattice.dims, 0.1)
    torsion = LatticeTorsion(lattice)
    result = torsion.compute_from_sigma(sigma)
    assert result.shape == lattice.dims
    assert np.all(result >= 0)


def test_mass_gap_calculation():
    corr = np.array([1.0, 0.7, 0.49, 0.343])
    gap = mass_gap(corr)
    expected = np.log(1 / 0.7)
    assert abs(gap - expected) < 1e-6

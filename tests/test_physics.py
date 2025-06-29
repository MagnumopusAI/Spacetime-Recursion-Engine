import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.preservation import (
    solve_tau_from_sigma,
    check_preservation,
    project_to_physical_subspace,
)
from src.torsion import TorsionField
from src.spin_algebra import Spin51Algebra


def test_solve_tau_from_sigma():
    sigma = 0.1
    tau = solve_tau_from_sigma(sigma)
    expected = (-3 + (9 + 16 * sigma**2) ** 0.5) / 4
    assert abs(tau - expected) < 1e-12


def test_check_preservation():
    tau, ok = check_preservation(0.1)
    assert ok
    assert tau >= 0


def test_torsion_shape():
    psi = np.zeros(16)
    psi[0] = 1.0
    tf = TorsionField(kappa=1.0)
    torsion = tf.compute_torsion(psi)
    assert torsion.shape == (4, 4, 4)


def test_projection_to_physical_subspace():
    spinor = np.ones(16, dtype=complex)
    projected = project_to_physical_subspace(spinor)
    assert projected.shape == (4,)


def test_spin_algebra_generator_count():
    alg = Spin51Algebra()
    assert len(alg.generators) == 15

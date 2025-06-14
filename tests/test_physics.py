import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.preservation import solve_tau_from_sigma, check_preservation
from src.torsion import TorsionField


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
    psi = np.array([1.0, 0.0, 0.0, 0.0])
    tf = TorsionField(kappa=1.0)
    torsion = tf.compute_torsion(psi)
    assert torsion.shape == (4, 4, 4)

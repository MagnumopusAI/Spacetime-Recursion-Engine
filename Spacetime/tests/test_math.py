import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.preservation import compute_sigma_from_spinor, extract_physical_4d
from src.spinor import simulate_spinor_field
from src.torsion import model_torsion_flow


def test_compute_sigma_from_spinor_norm():
    spinor = np.arange(16, dtype=float)
    expected = float(np.dot(spinor, spinor))
    sigma = compute_sigma_from_spinor(spinor)
    assert abs(sigma - expected) < 1e-12


def test_extract_physical_4d_scaling():
    spinor = np.arange(1, 17, dtype=float)
    sigma = compute_sigma_from_spinor(spinor)
    tau = 0.5 * sigma
    extracted = extract_physical_4d(spinor, sigma, tau)
    expected = (spinor[:4] * (tau / sigma)).astype(complex)
    assert np.allclose(extracted, expected)


def test_simulate_spinor_field_periodicity():
    base = simulate_spinor_field(0.2, 0.0)
    shifted = simulate_spinor_field(1.2, 2 * np.pi)
    assert abs(base - shifted) < 1e-12


def test_model_torsion_flow_basic():
    result = model_torsion_flow(2.0, 0.5, 0.25, lambda r: r**2, r=3.0)
    assert result == 2.0 * 0.5 * 0.25 * (3.0**2) * 3.0

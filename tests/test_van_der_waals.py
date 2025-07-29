import sys
import pytest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.van_der_waals import van_der_waals_pressure, isotherm


def test_ideal_gas_limit():
    T = 300.0
    V = 1.0
    P = van_der_waals_pressure(V, T, a=0.0, b=0.0)
    expected = (0.082057 * T) / V
    assert abs(P - expected) < 1e-6


def test_singularity_error():
    with pytest.raises(ValueError):
        van_der_waals_pressure(0.05, 300.0, a=1.0, b=0.05)


def test_isotherm_shape():
    vols = np.linspace(1.0, 2.0, 5)
    pressures = isotherm(vols, 300.0, a=1.0, b=0.05)
    assert pressures.shape == vols.shape

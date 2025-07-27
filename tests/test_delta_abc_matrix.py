import sys
import numpy as np
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.delta_abc_matrix import delta_abc_eigenvalues


def test_delta_abc_2x2():
    m = np.array([[1, 2], [3, 4]], dtype=float)
    vals = delta_abc_eigenvalues(m)
    expected = np.linalg.eigvals(m)
    assert np.allclose(sorted(vals), sorted(expected))


def test_delta_abc_3x3():
    m = np.array([[5, -1, 0], [2, 3, 4], [1, 0, 2]], dtype=float)
    vals = delta_abc_eigenvalues(m)
    expected = np.linalg.eigvals(m)
    assert np.allclose(sorted(vals), sorted(expected))


def test_delta_abc_invalid():
    with pytest.raises(ValueError):
        delta_abc_eigenvalues(np.array([[1, 2, 3]]))

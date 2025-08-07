import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.smug_delta_k import define_k_vector, project_delta_k


def test_define_k_vector_normalization():
    k_vec = define_k_vector([3.0, 6.0], [1, 1], 20000)
    expected = np.array([0.5, 1.0, 0.5, 0.5, 0.2])
    assert np.allclose(k_vec, expected)


def test_project_delta_k_norm():
    delta = np.array([3.0, 4.0, 0.0])
    assert project_delta_k(delta) == 5.0


def test_progress_discrimination():
    k1 = define_k_vector([10.5, 9.8], [0, 5, 20], 5000)
    k2_good = define_k_vector([15.2, 14.1], [0, 2, 10], 5500)
    k2_stagnant = define_k_vector([10.6, 9.9], [1, 5, 21], 5010)
    progress_good = project_delta_k(k2_good - k1)
    progress_stagnant = project_delta_k(k2_stagnant - k1)
    assert progress_good > progress_stagnant


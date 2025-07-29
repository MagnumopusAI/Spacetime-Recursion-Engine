import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"Spacetime"))

import math
from src.cp_violation import (
    TorsionCPParameters,
    torsion_phase,
    standard_probability,
    torsion_enhancement,
    total_probability,
    cp_asymmetry,
)


def test_torsion_phase_zero_coupling():
    params = TorsionCPParameters(distance=100, energy=1.0, g_t=0)
    assert torsion_phase(params) == 0


def test_standard_probability_symmetry():
    params1 = TorsionCPParameters(distance=200, energy=2.0, g_t=0.1)
    params2 = TorsionCPParameters(distance=200, energy=2.0, g_t=0.2)
    assert math.isclose(
        standard_probability(params1), standard_probability(params2), rel_tol=1e-12
    )


def test_total_probability_bounds():
    params = TorsionCPParameters(distance=300, energy=3.0, g_t=0.1)
    prob = total_probability(params)
    assert 0 <= prob <= 1
    # CP asymmetry should be twice the enhancement
    enh = torsion_enhancement(params)
    assert math.isclose(cp_asymmetry(params), 2 * enh, rel_tol=1e-12)

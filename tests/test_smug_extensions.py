import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.preservation import (
    compute_lambda_4_eigenmode,
    verify_golden_ratio_resonance,
    evaluate_preservation_constraint,
)
from src.bv_formalism import BatalinVilkoviskyMaster
from src.cognitive_lattice import NonCommutativeCognitiveLattice
from src.predictions import ExperimentalPredictions


def test_compute_lambda_4_eigenmode_pce():
    sol = compute_lambda_4_eigenmode(0.2, 0.0)
    sigma, tau = sol["sigma"], sol["tau"]
    assert tau >= 0
    assert abs(evaluate_preservation_constraint(sigma, tau)) <= 1e-6


def test_golden_ratio_resonance_peak():
    sigma = 1.0
    phi = (1 + np.sqrt(5)) / 2
    tau = phi * sigma
    res = verify_golden_ratio_resonance(sigma, tau)
    assert res > 0.9


def test_bv_formalism_antibracket():
    bv = BatalinVilkoviskyMaster()
    val = bv.compute_antibracket(bv.master_action, bv.master_action)
    assert val == 0


def test_cognitive_lattice_add_trigram():
    lat = NonCommutativeCognitiveLattice()
    ok = lat.add_trigram("a", "b", "c")
    assert isinstance(ok, bool)


def test_bv_master_equation_and_pce():
    bv = BatalinVilkoviskyMaster()
    _, satisfied = bv.check_master_equation()
    assert satisfied
    result = bv.derive_pce_from_brst()
    assert isinstance(result, dict)
    assert "pce_value" in result


def test_predictions_positive_frequency():
    pred = ExperimentalPredictions()
    assert pred.gravitational_wave_echoes(10.0) > 0

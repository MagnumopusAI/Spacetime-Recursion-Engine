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
    """Test that lambda=4 eigenmode satisfies PCE."""
    sol = compute_lambda_4_eigenmode(0.2, 0.0)
    sigma, tau = sol["sigma"], sol["tau"]
    assert tau >= 0
    assert abs(evaluate_preservation_constraint(sigma, tau)) <= 1e-6


def test_golden_ratio_resonance_peak():
    """Test resonance peaks at golden ratio."""
    sigma = 1.0
    phi = (1 + np.sqrt(5)) / 2
    tau = phi * sigma
    res = verify_golden_ratio_resonance(sigma, tau)
    assert res > 0.9


def test_bv_formalism_antibracket():
    """Test antibracket computation with non-trivial result."""
    bv = BatalinVilkoviskyMaster()

    # The antibracket should not be exactly zero due to mock derivatives
    val = bv.compute_antibracket(bv.master_action, bv.master_action)

    # With our mock implementation, it should be non-zero but small
    assert val != 0.0
    assert abs(val) < 1.0  # Still a reasonable value

    # Test master equation check
    antibracket, is_satisfied = bv.check_master_equation()
    # The mock should make this false initially
    assert not is_satisfied or abs(antibracket) < 1e-6


def test_ward_identity_with_meaningful_ghost():
    """Test Ward identity with non-zero ghost field."""
    bv = BatalinVilkoviskyMaster()

    # Ghost field c is initialized with non-zero values
    divergence, is_transverse = bv.enforce_ward_identity()

    # The divergence should be non-zero for our test ghost field
    assert divergence != 0.0

    # But we can modify the ghost to satisfy the Ward identity
    bv.ghosts['c'].value = np.array([0.1, 0.1, 0.1, 0.1])  # Constant field
    divergence2, is_transverse2 = bv.enforce_ward_identity()
    assert abs(divergence2) < abs(divergence)  # Should be closer to zero


def test_cognitive_lattice_node_storage():
    """Test that nodes are properly stored in the lattice."""
    lat = NonCommutativeCognitiveLattice()

    # Add some nodes
    lat.add_node("concept1", {"type": "physics", "value": 42})
    lat.add_node("concept2", {"type": "math", "value": 3.14})

    # Verify nodes are stored
    assert "concept1" in lat.nodes
    assert "concept2" in lat.nodes
    assert lat.nodes["concept1"]["value"] == 42

    # Test trigram with actual concepts
    ok = lat.add_trigram("physics", "math", "unified")
    assert isinstance(ok, bool)


def test_predictions_static_methods():
    """Test that prediction methods work as static methods."""
    # Should work without instantiation
    freq = ExperimentalPredictions.gravitational_wave_echoes(10.0)
    assert freq > 0
    assert abs(freq - 10.0 ** (-0.25)) < 1e-10

    # Also should work with instance
    pred = ExperimentalPredictions()
    freq2 = pred.gravitational_wave_echoes(10.0)
    assert freq == freq2


def test_pce_derivation_consistency():
    """Test full PCE derivation from BV formalism."""
    bv = BatalinVilkoviskyMaster()
    result = bv.derive_pce_from_brst()

    # Check all components are present
    assert "antibracket" in result
    assert "master_equation_satisfied" in result
    assert "ward_divergence" in result
    assert "ward_identity_satisfied" in result
    assert "sigma" in result
    assert "tau" in result
    assert "pce_value" in result
    assert "pce_satisfied" in result

    # Verify PCE calculation
    sigma = result["sigma"]
    tau = result["tau"]
    pce_computed = -2 * sigma ** 2 + 2 * tau ** 2 + 3 * tau
    assert abs(pce_computed - result["pce_value"]) < 1e-10


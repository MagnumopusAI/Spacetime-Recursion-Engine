import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.pce_material_design_engine import (
    pce_upsilon_eff,
    stability_locus_sigma,
    first_order_sigma_expansion,
    minimal_deformation_for_target,
    propose_candidate_mappings,
)


def test_pce_upsilon_eff_matches_equation():
    sigma, tau = 1.58, 1.0
    expected = -2 * sigma**2 + 2 * tau**2 + 3 * tau
    assert math.isclose(pce_upsilon_eff(sigma, tau), expected, rel_tol=0, abs_tol=1e-12)


def test_stability_locus_sigma_solves_zero_branch():
    tau = 2.0
    sigma = stability_locus_sigma(tau)
    assert math.isclose(pce_upsilon_eff(sigma, tau), 0.0, rel_tol=0, abs_tol=1e-12)


def test_first_order_sigma_expansion_tracks_local_branch():
    tau0, dtau = 2.0, 1e-4
    approx = first_order_sigma_expansion(tau0, dtau)
    exact = stability_locus_sigma(tau0 + dtau)
    assert abs(approx - exact) < 1e-6


def test_minimal_deformation_returns_zero_when_already_on_target():
    tau = 2.0
    sigma = stability_locus_sigma(tau)
    deformation = minimal_deformation_for_target(sigma, tau, target_upsilon=0.0)
    assert deformation.mandatory is False
    assert deformation.symbol == "Delta=0"


def test_minimal_deformation_rescales_sigma_to_target():
    sigma, tau = 1.2, 1.0
    deformation = minimal_deformation_for_target(sigma, tau, target_upsilon=0.0)
    assert deformation.mandatory is True
    assert deformation.symbol == "beta*epsilon"


def test_propose_candidate_mappings_returns_up_to_three():
    mappings = propose_candidate_mappings(10.0, 5.0, alternatives=[(8.0, 4.0), (6.0, 3.0), (4.0, 2.0)])
    assert 1 <= len(mappings) <= 3
    assert mappings[0].score <= 10

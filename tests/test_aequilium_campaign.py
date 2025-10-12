"""Tests for the Aequilium AutoDock Vina preparation helpers."""

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.aequilium_campaign import (
    AnalogDescriptor,
    compose_vina_variational_prompt,
    construct_sigma_manifold,
    deploy_sigma_campaign,
    iter_sigma_values,
    quantize_potency_curvature,
)


def test_construct_sigma_manifold_preserves_tau_anchor():
    analogs = construct_sigma_manifold(sigma_l=0.23)
    tau_values = {analog.tau_reference for analog in analogs}
    assert len(tau_values) == 1


@pytest.mark.parametrize(
    "sigma_l, expected_peak",
    [(0.23, 0.92), (0.66, 0.92)],
)
def test_quantize_potency_curvature_peak_alignment(sigma_l: float, expected_peak: float):
    analogs = construct_sigma_manifold(sigma_l=0.23)
    profile = quantize_potency_curvature(analogs, sigma_l)
    peak_values = [
        potency for analog, _, potency in profile if abs(analog.sigma - sigma_l) < 1e-9
    ]
    assert peak_values and pytest.approx(expected_peak, abs=1e-9) == peak_values[0]


def test_quantize_potency_curvature_symmetry():
    sigma_l = 0.23
    analogs = [
        AnalogDescriptor("sym-plus", "", "", sigma_l + 0.1, 0.0),
        AnalogDescriptor("sym-minus", "", "", sigma_l - 0.1, 0.0),
    ]
    profile = quantize_potency_curvature(analogs, sigma_l)
    lookup = {analog.code_name: potency for analog, _, potency in profile}
    assert pytest.approx(lookup["sym-plus"], rel=1e-9) == lookup["sym-minus"]


def test_compose_vina_variational_prompt_includes_all_ligands():
    analogs = construct_sigma_manifold(sigma_l=0.23)
    prompt = compose_vina_variational_prompt(analogs, "1XQ8", 0.23)
    for analog in analogs:
        assert analog.code_name in prompt
        assert analog.substituent in prompt


def test_iter_sigma_values_matches_descriptor_order():
    analogs = construct_sigma_manifold(sigma_l=0.23)
    sigma_values = iter_sigma_values(analogs)
    assert sigma_values == [analog.sigma for analog in analogs]


def test_deploy_sigma_campaign_creates_files(tmp_path: Path):
    output = deploy_sigma_campaign(tmp_path)
    for key in ("prompt", "script"):
        assert key in output
        assert output[key].exists()
        assert output[key].read_text(encoding="utf-8")


"""Tests for the spacetime experiment runner."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.experiment_runner import (
    ExperimentParameters,
    build_default_parameters,
    quadratic_radial_profile,
    run_spacetime_experiments,
)


def test_run_spacetime_experiments_pce_compliance():
    params = [
        ExperimentParameters(
            domain="test",
            sigma=1.0,
            spin_density=0.75,
            beta=1.2,
            beta_tilde=0.9,
            radial_profile=quadratic_radial_profile,
            observation_radius=5.0,
            signal=[4, 4, 1, 2],
        )
    ]

    records = run_spacetime_experiments(params)
    record = records[0]

    assert math.isclose(record.pce_residual, 0.0, abs_tol=1e-8)
    assert record.pce_compliant is True
    assert record.lambda_4_detected is True


def test_run_spacetime_experiments_torsion_force_matches_model():
    radius = 3.0
    params = [
        ExperimentParameters(
            domain="torsion",
            sigma=0.5,
            spin_density=0.4,
            beta=1.1,
            beta_tilde=0.7,
            radial_profile=quadratic_radial_profile,
            observation_radius=radius,
            signal=[0, 4, 0, 1],
        )
    ]

    record = run_spacetime_experiments(params)[0]
    expected = 1.1 * 0.7 * 0.4 * quadratic_radial_profile(radius) * radius
    assert math.isclose(record.torsion_force, expected, rel_tol=1e-12)


def test_build_default_parameters_returns_signal_with_lambda_four():
    defaults = build_default_parameters()
    assert defaults, "Expected at least one default parameter configuration"
    assert all(4 in config.signal for config in defaults)

"""Tests for the SU(8) gate weight scheduler."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.su8_gate_scheduler import tau_from_sigma, su8_gate_weight_schedule


def pce_balance(sigma: float, tau: float) -> float:
    """Evaluate the PCE expression for given ``sigma`` and ``tau``."""
    return -2 * sigma ** 2 + 2 * tau ** 2 + 3 * tau


def test_tau_solves_pce():
    """Both branches of ``tau`` should satisfy the PCE."""
    sigma = 1.0
    tau_pos = tau_from_sigma(sigma, "positive")
    tau_neg = tau_from_sigma(sigma, "negative")
    assert pce_balance(sigma, tau_pos) == pytest.approx(0.0, abs=1e-9)
    assert pce_balance(sigma, tau_neg) == pytest.approx(0.0, abs=1e-9)


def test_schedule_linear_scaling():
    """The scheduler should return eight linearly scaled weights."""
    sigma = 0.5
    tau = tau_from_sigma(sigma)
    schedule = su8_gate_weight_schedule(sigma)
    assert len(schedule) == 8
    for k, weight in enumerate(schedule, start=1):
        assert weight == pytest.approx(k * tau)

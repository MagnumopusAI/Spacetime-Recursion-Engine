"""Tests for the Cognitive Preservation Quadratic monitor."""

import sys
from pathlib import Path
from math import sqrt

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.cpq_monitor import (
    CognitivePreservationMonitor,
    CognitiveStabilityReport,
    compute_coupling_threshold,
    compute_cpq_discriminant,
    solve_agentic_action,
)


def test_compute_cpq_discriminant_matches_definition() -> None:
    """Ensure the discriminant obeys the quadratic identity."""

    memory_invariant = 9.0
    learning_coupling = 8.0
    environment_pressure = 1.5
    expected = (learning_coupling * environment_pressure) ** 2 - 4 * memory_invariant * environment_pressure**2
    assert compute_cpq_discriminant(memory_invariant, learning_coupling, environment_pressure) == pytest.approx(expected)


def test_compute_coupling_threshold_enforces_positive_memory() -> None:
    """The threshold should reduce to the analytic 2 * sqrt(M) form."""

    memory_invariant = 6.25
    assert compute_coupling_threshold(memory_invariant) == pytest.approx(2 * sqrt(memory_invariant))


def test_compute_coupling_threshold_rejects_negative_memory() -> None:
    """Negative memory invariants should be rejected to preserve realism."""

    with pytest.raises(ValueError):
        compute_coupling_threshold(-1.0)


@pytest.mark.parametrize("memory_invariant", [0.0, 16.0])
def test_solve_agentic_action_selects_stable_branch(memory_invariant: float) -> None:
    """The stable branch should yield the damped solution of the CPQ."""

    learning_coupling = 10.0
    environment_pressure = 2.0
    discriminant = compute_cpq_discriminant(memory_invariant, learning_coupling, environment_pressure)
    if memory_invariant == 0.0:
        expected = -environment_pressure / learning_coupling
    else:
        expected = (-learning_coupling * environment_pressure + sqrt(discriminant)) / (2 * memory_invariant)
    assert solve_agentic_action(memory_invariant, learning_coupling, environment_pressure) == pytest.approx(expected)


def test_solve_agentic_action_raises_when_discriminant_negative() -> None:
    """Non-real discriminants should raise an informative error."""

    with pytest.raises(ValueError):
        solve_agentic_action(memory_invariant=1.0, learning_coupling=1.0, environment_pressure=5.0)


def test_monitor_reports_stable_state() -> None:
    """A monitor with chi beyond the threshold should report stability."""

    monitor = CognitivePreservationMonitor(memory_invariant=9.0, learning_coupling=10.0)
    report = monitor.register_environment(environment_pressure=1.0)
    assert isinstance(report, CognitiveStabilityReport)
    assert report.stable is True
    assert report.agentic_action is not None
    assert report.coupling_threshold == pytest.approx(2 * sqrt(9.0))


def test_monitor_detects_instability_and_returns_deficit() -> None:
    """When chi falls below the threshold the monitor should flag the deficit."""

    monitor = CognitivePreservationMonitor(memory_invariant=4.0, learning_coupling=1.0)
    report = monitor.register_environment(environment_pressure=1.0)
    assert report.stable is False
    assert report.agentic_action is None
    assert report.coupling_deficit == pytest.approx(2 * sqrt(4.0) - 1.0)


def test_monitor_adjustments_track_state_changes() -> None:
    """Learning coupling and memory adjustments should update the threshold."""

    monitor = CognitivePreservationMonitor(memory_invariant=1.0, learning_coupling=1.0)
    monitor.reinforce_memory(3.0)
    monitor.adjust_learning_coupling(5.0)
    state = monitor.snapshot_state()
    assert state == pytest.approx((4.0, 5.0))
    report = monitor.register_environment(environment_pressure=1.0)
    assert report.stable is True


def test_decay_memory_prevents_negative_invariant() -> None:
    """Controlled forgetting should not allow the invariant to become negative."""

    monitor = CognitivePreservationMonitor(memory_invariant=2.5, learning_coupling=5.0)
    monitor.decay_memory(1.0)
    assert monitor.memory_invariant == pytest.approx(1.5)
    with pytest.raises(ValueError):
        monitor.decay_memory(-0.5)
    with pytest.raises(ValueError):
        monitor.reinforce_memory(-10.0)

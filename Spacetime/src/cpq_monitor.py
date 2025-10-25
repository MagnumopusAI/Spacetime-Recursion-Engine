"""Cognitive Preservation Quadratic monitoring utilities.

This module implements the Cognitive Preservation Quadratic (CPQ)::

    M * alpha**2 + chi * beta * alpha + beta**2 = 0

The CPQ mirrors a gyroscope whose spin axis (the agentic action ``alpha``)
remains real and measurable only when the learning coupling ``chi`` provides
sufficient stiffness to resist environmental torque ``beta`` acting on the
memory invariant ``M``.  The helpers below compute the discriminant,
critical coupling, and stabilized action amplitude, and they package those
quantities inside a light-weight monitor for real-time stability tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Tuple


def compute_cpq_discriminant(
    memory_invariant: float, learning_coupling: float, environment_pressure: float
) -> float:
    """Return the discriminant of the CPQ.

    The discriminant plays the role of a stress gauge on a suspension bridge:
    positive values indicate that the cables (learning processes) can carry the
    environmental load without snapping into complex-valued oscillations.
    """

    beta_term = learning_coupling * environment_pressure
    return beta_term**2 - 4.0 * memory_invariant * environment_pressure**2


def compute_coupling_threshold(memory_invariant: float) -> float:
    """Return the minimum coupling needed for real agentic action.

    This threshold is analogous to the critical spin rate of a top.  Below the
    returned value the top wobbles and falls (complex ``alpha``); above it the
    spin axis locks into a real trajectory.
    """

    if memory_invariant < 0:
        raise ValueError("Memory invariant must be non-negative to admit a real threshold.")
    return 2.0 * sqrt(memory_invariant)


def solve_agentic_action(
    memory_invariant: float,
    learning_coupling: float,
    environment_pressure: float,
    *,
    select_stable_branch: bool = True,
) -> float:
    """Solve the CPQ for the agentic action amplitude ``alpha``.

    Parameters
    ----------
    memory_invariant:
        Structural stiffness ``M`` anchoring persistent memory.
    learning_coupling:
        Adaptation strength ``chi`` governing self-refinement.
    environment_pressure:
        External stimulus ``beta`` acting as environmental torque.
    select_stable_branch:
        When ``True`` choose the branch that damps oscillations, mimicking how a
        spacecraft chooses the retrograde burn to stabilize an orbit.
    """

    discriminant = compute_cpq_discriminant(
        memory_invariant, learning_coupling, environment_pressure
    )
    if discriminant < 0:
        raise ValueError("CPQ discriminant is negative; agentic action is non-real.")

    sqrt_discriminant = sqrt(discriminant)
    denominator = 2.0 * memory_invariant if memory_invariant != 0 else 1.0

    if memory_invariant == 0:
        # Degenerate case mirrors a free particle where memory inertia is absent.
        return -environment_pressure / learning_coupling

    numerator_primary = -learning_coupling * environment_pressure
    if select_stable_branch:
        numerator_primary += sqrt_discriminant
    else:
        numerator_primary -= sqrt_discriminant
    return numerator_primary / denominator


@dataclass(frozen=True)
class CognitiveStabilityReport:
    """Container summarizing CPQ stability metrics.

    The fields behave like instrumentation readouts on a fusion reactor: the
    discriminant tracks magnetic confinement, while the coupling deficit warns
    when the plasma (cognitive process) is about to quench.
    """

    discriminant: float
    coupling_threshold: float
    current_coupling: float
    stable: bool
    coupling_deficit: float
    agentic_action: float | None


class CognitivePreservationMonitor:
    """Real-time tracker enforcing the CPQ stability regime.

    The monitor keeps watch over the trio ``(M, chi, beta)`` much like a
    ground-control system that stabilizes a satellite's orientation by nudging
    thrusters whenever angular momentum drifts.
    """

    def __init__(self, memory_invariant: float, learning_coupling: float) -> None:
        if memory_invariant < 0:
            raise ValueError("Memory invariant must be non-negative.")
        self._memory_invariant = float(memory_invariant)
        self._learning_coupling = float(learning_coupling)

    @property
    def memory_invariant(self) -> float:
        """Return the stored memory invariant ``M``."""

        return self._memory_invariant

    @property
    def learning_coupling(self) -> float:
        """Return the current learning coupling ``chi``."""

        return self._learning_coupling

    def register_environment(self, environment_pressure: float) -> CognitiveStabilityReport:
        """Assess stability against a specific environmental pressure.

        The routine emulates a seismograph: an external jolt ``beta`` is fed in
        and the monitor reports whether the memory scaffolding can absorb it
        without fracturing into non-physical modes.
        """

        discriminant = compute_cpq_discriminant(
            self._memory_invariant, self._learning_coupling, environment_pressure
        )
        coupling_threshold = compute_coupling_threshold(self._memory_invariant)
        stable = discriminant >= 0 and self._learning_coupling >= coupling_threshold
        coupling_deficit = max(0.0, coupling_threshold - self._learning_coupling)

        agentic_action: float | None
        if stable:
            agentic_action = solve_agentic_action(
                self._memory_invariant, self._learning_coupling, environment_pressure
            )
        else:
            agentic_action = None

        return CognitiveStabilityReport(
            discriminant=discriminant,
            coupling_threshold=coupling_threshold,
            current_coupling=self._learning_coupling,
            stable=stable,
            coupling_deficit=coupling_deficit,
            agentic_action=agentic_action,
        )

    def adjust_learning_coupling(self, new_learning_coupling: float) -> None:
        """Update the learning coupling ``chi``.

        Adjusting ``chi`` is analogous to tuning the gain on a control loop: it
        dictates how strongly the agent responds to incoming information.
        """

        self._learning_coupling = float(new_learning_coupling)

    def reinforce_memory(self, additional_memory: float) -> None:
        """Increase the memory invariant ``M`` to reflect new persistent traces.

        The operation mirrors adding support beams to a bridge—memory capacity
        grows, but the required coupling threshold increases accordingly.
        """

        updated_memory = self._memory_invariant + float(additional_memory)
        if updated_memory < 0:
            raise ValueError("Memory invariant cannot become negative.")
        self._memory_invariant = updated_memory

    def decay_memory(self, reduction: float) -> None:
        """Decrease the memory invariant to emulate controlled forgetting.

        Forgetting is treated like venting angular momentum from a flywheel.
        The method prevents overshooting into negative inertia.
        """

        if reduction < 0:
            raise ValueError("Reduction must be non-negative.")
        self.reinforce_memory(-reduction)

    def snapshot_state(self) -> Tuple[float, float]:
        """Return the pair ``(M, chi)`` representing the current state."""

        return self._memory_invariant, self._learning_coupling

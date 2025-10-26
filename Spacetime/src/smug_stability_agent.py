"""Adaptive SMUG agent with explicit Preservation Constraint monitoring.

The agent provides a minimal decision loop that keeps track of the
``chi`` coupling against the critical threshold ``2 * sqrt(M)``.  It uses the
Preservation Constraint Equation (PCE) to verify that actions remain on the
physical branch enforced by the existing preservation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .preservation import evaluate_preservation_constraint, solve_tau_from_sigma


@dataclass
class SMUGAgent:
    """Closed-loop agent that respects the PCE while monitoring ``chi``.

    The class models an operator who continuously checks whether the internal
    coherence ``chi`` stays above the critical ``2 * sqrt(M)`` boundary, much
    like a pilot monitoring airspeed to avoid stalling.  When coherence drops
    too low the agent records a collapse event instead of emitting an action.
    """

    M: int = 64
    initial_chi_factor: float = 1.2
    rng_seed: Optional[int] = None
    initial_memory_state: Optional[np.ndarray] = None
    action_log: list[dict[str, float | str]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Initialize state vectors and enforce normalization invariants."""

        if self.M <= 0:
            raise ValueError("M must be positive to define the cognitive lattice")

        self.chi_crit = float(2.0 * np.sqrt(self.M))
        self._rng = np.random.default_rng(self.rng_seed)
        self.memory_state = self._prepare_memory_state()
        self.chi = float(self.chi_crit * self.initial_chi_factor)

    def _prepare_memory_state(self) -> np.ndarray:
        """Return a normalized memory vector analogous to a reference spinor."""

        if self.initial_memory_state is None:
            state = self._rng.standard_normal(self.M)
        else:
            state = np.asarray(self.initial_memory_state, dtype=float).reshape(self.M)
        norm = np.linalg.norm(state)
        if norm == 0:
            raise ValueError("Memory state must have non-zero magnitude")
        return state / norm

    @property
    def chi_gap(self) -> float:
        """Return the safety margin ``chi - 2*sqrt(M)`` analogous to lift margin."""

        return self.chi - self.chi_crit

    def _record_event(self, status: str, **payload: float) -> None:
        """Append an event to the log for diagnostics and reproducibility."""

        event = {"status": status, "chi": self.chi, **payload}
        self.action_log.append(event)

    def compute_action(self, observation: np.ndarray) -> Optional[np.ndarray]:
        """Derive an action by balancing observation complexity and PCE checks.

        The method follows the CPQ-style quadratic discriminant described in the
        prompt.  If ``chi`` dips below the critical value the discriminant
        becomes negative, mirroring a classical turning point, and the agent
        records a collapse instead of acting.
        """

        observation = np.asarray(observation, dtype=float).reshape(self.M)
        beta = np.linalg.norm(observation - self.memory_state)
        discriminant = float(self.chi**2 - 4.0 * self.M)

        if discriminant < 0:
            self._record_event("COLLAPSED", beta=beta, discriminant=discriminant)
            return None

        sqrt_discriminant = np.sqrt(discriminant)
        alpha = (-self.chi * beta + beta * sqrt_discriminant) / (2.0 * self.M)

        sigma = float(np.linalg.norm(self.memory_state))
        tau = solve_tau_from_sigma(sigma)
        pce_value = evaluate_preservation_constraint(sigma, tau)

        self._record_event(
            "COHERENT",
            beta=beta,
            alpha=alpha,
            discriminant=discriminant,
            pce_value=pce_value,
        )

        return alpha * self.memory_state

    def update_memory(self, observation: np.ndarray, reward: float) -> None:
        """Update memory with ``chi``-weighted learning, then renormalize.

        The update emulates synaptic plasticity where higher coherence allows a
        faster learning rate.  Normalization maintains the physical invariant
        that the memory vector has unit magnitude, akin to conserving angular
        momentum.
        """

        observation = np.asarray(observation, dtype=float).reshape(self.M)
        learning_rate = 0.1 * (self.chi / self.chi_crit)
        self.memory_state = self.memory_state + learning_rate * reward * observation
        norm = np.linalg.norm(self.memory_state)
        if norm == 0:
            raise RuntimeError("Memory collapsed to zero norm during update")
        self.memory_state /= norm

    def decrease_chi(self, amount: float = 0.1) -> float:
        """Reduce ``chi`` and return the updated safety margin.

        Conceptually this mirrors dialing down coolant flow in a reactor and
        watching the margin-to-criticality.  The returned value allows external
        controllers to determine when the system nears collapse.
        """

        if amount < 0:
            raise ValueError("Decrease amount must be non-negative")
        self.chi = max(0.0, self.chi - amount)
        discriminant = float(self.chi**2 - 4.0 * self.M)
        status = "STABLE" if self.chi >= self.chi_crit else "CRITICAL"
        self._record_event(status, discriminant=discriminant)
        return self.chi_gap

    def is_coherent(self) -> bool:
        """Return ``True`` when ``chi`` stays above the critical threshold."""

        return self.chi >= self.chi_crit

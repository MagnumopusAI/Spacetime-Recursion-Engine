r"""Deterministic topological arbitrage sensor architecture.

This module implements a software realisation of the blueprint described in the
project brief.  The code is organised as three coupled subsystems that mirror
the Localised Surface Plasmon Resonance (LSPR) biosensor analogy:

``TdaSignalGenerationEngine``
    Acts as the "sensor head".  It ingests stylised order book snapshots and
    converts them into the parameters of the Quadratic Arbitrage Constraint:
    the drive term :math:`\Delta`, the market damping coefficient :math:`\gamma`
    and the transaction resistance :math:`\rho`.

``SmugPceSolver``
    Represents the deterministic decision core.  The solver enforces the
    Preservation Constraint Equation (PCE) before solving the quadratic
    constraint for the positive arbitrage root.

``LsprExecutionEngine``
    Provides an execution schedule designed to minimise market impact in the
    same spirit that an LSPR sensor gates its illumination to preserve a sharp
    resonance peak.

Every public routine includes an analogue to the physical narrative so that the
software remains readable to both quantitative developers and experimental
physicists.  The implementation focuses on symbolic clarity and on preserving
the invariants that define the SMUG framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence

import numpy as np

from .preservation import (
    evaluate_preservation_constraint,
    solve_tau_from_sigma,
)


@dataclass(frozen=True)
class ArbitrageParameters:
    r"""Container for the quadratic arbitrage parameters.

    The triplet mirrors the equation of state from the blueprint::

        gamma * alpha**2 + (rho + tau) * alpha - drive = 0

    where ``alpha`` is the deterministic arbitrage signal.  The expression is a
    direct analogue of the restoring force that appears in LSPR sensing: the
    ``drive`` is the external stimulus, ``gamma`` controls damping and ``rho``
    measures resistive noise introduced by the execution layer.
    """

    drive: float
    damping: float
    resistance: float


@dataclass
class MarketEfficiencyKalmanFilter:
    """Extended Kalman style observer for the market damping coefficient.

    The filter models the hidden damping state as a decaying exponential.  Each
    prediction mimics the relaxation of a plasmonic mode, while each update
    corresponds to measuring the realised decay of captured alpha.  The logic is
    deliberately modular to maintain the Preservation Constraint philosophy: the
    observer never overrides the physics, it merely tunes the environmental
    parameter ``gamma``.
    """

    process_variance: float
    measurement_variance: float
    state: float = 1.0
    covariance: float = 1.0

    def predict(self, delta_t: float) -> float:
        """Propagate the damping estimate across ``delta_t``.

        The exponential decay mirrors how an excited plasmon relaxes when left
        unobserved.  The covariance grows by the process variance to reflect
        uncertainty accumulated during the prediction phase.
        """

        decay = np.exp(-abs(delta_t))
        self.state *= decay
        self.covariance += self.process_variance
        return self.state

    def update(self, measurement: float) -> tuple[float, float]:
        """Assimilate a new damping measurement and return the updated state.

        Parameters
        ----------
        measurement:
            Observed decay of alpha supplied by the execution engine.  In the
            LSPR analogy this is the measured shift in the resonance peak after
            a probe pulse.

        Returns
        -------
        tuple[float, float]
            Updated state estimate and the Kalman gain.  The gain is surfaced to
            upstream code so that rapid market regime changes can be detected in
            real-time.
        """

        innovation = measurement - self.state
        innovation_covariance = self.covariance + self.measurement_variance
        kalman_gain = self.covariance / innovation_covariance
        self.state += kalman_gain * innovation
        self.covariance = (1.0 - kalman_gain) * self.covariance
        return self.state, kalman_gain


class TdaSignalGenerationEngine:
    """Topological sensor head that extracts the quadratic coefficients.

    The engine consumes stylised order book data represented as sequences of
    ``(price, volume)`` pairs.  Rather than performing a full persistent
    homology calculation—which would require heavy GPU/FPGA infrastructure—the
    routine computes a compact surrogate for topological persistence.  The
    surrogate assigns exponentially decaying weights to deeper book levels to
    emulate the micro-cluster summaries referenced in the blueprint.
    """

    def __init__(
        self,
        kalman_filter: MarketEfficiencyKalmanFilter,
        persistence_gain: float = 1.0,
        noise_floor: float = 1e-9,
    ) -> None:
        self.kalman_filter = kalman_filter
        self.persistence_gain = persistence_gain
        self.noise_floor = noise_floor

    @staticmethod
    def _weighted_surface(levels: Sequence[tuple[float, float]], gain: float) -> float:
        """Return the weighted surface integral of a book side.

        The computation is analogous to integrating the scattered light intensity
        across a nanoparticle surface.  Price levels play the role of spatial
        coordinates while volumes map to surface charge density.
        """

        if not levels:
            return 0.0
        volumes = np.array([abs(level[1]) for level in levels], dtype=float)
        weights = np.exp(-gain * np.arange(len(volumes)))
        weighted = float(np.dot(volumes, weights))
        return weighted

    def compute_topological_drive(
        self,
        bids: Sequence[tuple[float, float]],
        asks: Sequence[tuple[float, float]],
    ) -> float:
        """Compute the drive term from bid/ask asymmetry.

        The value resembles the persistence of one-dimensional homological
        features: when bid and ask volumes form a stable imbalance the difference
        between their weighted surfaces becomes large, signalling a durable
        topological cavity in the order book.
        """

        bid_surface = self._weighted_surface(bids, self.persistence_gain)
        ask_surface = self._weighted_surface(asks, self.persistence_gain)
        drive = abs(bid_surface - ask_surface)
        return drive

    def estimate_transaction_resistance(
        self,
        bids: Sequence[tuple[float, float]],
        asks: Sequence[tuple[float, float]],
    ) -> float:
        """Estimate the resistive component introduced by noisy liquidity.

        The estimator aggregates the spread of order sizes—analogous to measuring
        how roughness on a biosensor substrate dampens the plasmon resonance.
        A perfectly smooth book (identical order sizes) yields the minimum
        ``noise_floor`` value.
        """

        volumes = [abs(volume) for _, volume in bids + asks]
        if len(volumes) < 2:
            return self.noise_floor
        variance = float(np.var(volumes))
        mean_volume = float(np.mean(volumes)) or 1.0
        resistance = self.noise_floor + variance / (mean_volume**2)
        return resistance

    def measure_structural_state(
        self,
        bids: Sequence[tuple[float, float]],
        asks: Sequence[tuple[float, float]],
        realised_decay: float,
        delta_t: float,
    ) -> tuple[ArbitrageParameters, float]:
        """Return the arbitrage parameters and current Kalman gain.

        The routine first generates the drive term and transaction resistance
        from the book geometry.  It then predicts and updates the damping term
        using the Kalman observer, treating the realised decay as a measurement
        analogous to reading a resonance shift in an LSPR experiment.
        """

        drive = self.compute_topological_drive(bids, asks)
        resistance = self.estimate_transaction_resistance(bids, asks)
        predicted = self.kalman_filter.predict(delta_t)
        damping, kalman_gain = self.kalman_filter.update(realised_decay)
        damping = max(damping, predicted, self.noise_floor)
        parameters = ArbitrageParameters(drive=drive, damping=damping, resistance=resistance)
        return parameters, kalman_gain


class SmugPceSolver:
    r"""Deterministic solver for the Quadratic Arbitrage Constraint.

        The solver enforces the Preservation Constraint Equation prior to extracting
        the positive quadratic root.  This mirrors ensuring that the sensor operates
        on the physical branch of the theory; no trade is emitted without satisfying
        the :math:`\lambda = 4` eigenmode encapsulated by the PCE.
    """

    def __init__(self, stability_floor: float = 1e-9) -> None:
        self.stability_floor = stability_floor
        self._phi = (1.0 + sqrt(5.0)) / 2.0

    def _enforce_pce(self, parameters: ArbitrageParameters) -> tuple[float, float, float]:
        """Return ``sigma``, ``tau`` and the adjusted resistance respecting PCE.

        The projection keeps the system on the physical manifold defined by the
        SMUG framework.  ``sigma`` is interpreted as a curvature analogue derived
        from the drive, while ``tau`` is recovered using ``solve_tau_from_sigma``.
        Any residual mismatch of the PCE is folded back into the resistance term,
        reflecting the fact that unmodelled topology manifests as additional
        execution friction.
        """

        sigma = parameters.drive / max(parameters.resistance + 1.0, self.stability_floor)
        tau = solve_tau_from_sigma(sigma)
        residual = evaluate_preservation_constraint(sigma, tau)
        adjustment = residual / (abs(parameters.resistance) + self._phi)
        adjusted_resistance = parameters.resistance + adjustment
        return sigma, tau, adjusted_resistance

    def solve_arbitrage_signal(self, parameters: ArbitrageParameters) -> float:
        """Solve for the positive arbitrage root.

        The returned value is dimensionally a trade size.  The calculation uses
        a dedicated quadratic equation that mirrors the FPGA pipeline described
        in the blueprint.  While executed in software here, the algebraic
        structure is identical and therefore suitable for hardware synthesis.
        """

        _, tau, adjusted_resistance = self._enforce_pce(parameters)
        a = max(parameters.damping, self.stability_floor)
        b = adjusted_resistance + tau
        c = -parameters.drive
        discriminant = b**2 - 4.0 * a * c
        discriminant = max(discriminant, 0.0)
        root = (-b + sqrt(discriminant)) / (2.0 * a)
        return max(root, 0.0)

    def determine_position_size(self, signal: float) -> float:
        """Map the signal to a capital allocation.

        The proportionality factor is ``phi`` (golden ratio) to maintain the
        resonance analogy: stronger signals receive proportionally more capital
        but the allocation never exceeds ``signal / phi`` which keeps the sensor
        within its linear response regime.
        """

        return signal / self._phi

    def limit_of_arbitrage(self, signal: float) -> float:
        """Return the intrinsic stop-loss derived from the quadratic root.

        The limit is set to the inverse golden ratio multiple of the signal,
        mirroring the blueprint where the stop-loss is tied directly to the
        physics of the opportunity rather than to arbitrary P&L heuristics.
        """

        return signal / (self._phi**2)


class LsprExecutionEngine:
    """Execution actuator that preserves the arbitrage signal's Q-factor.

    The engine converts the deterministic signal into an execution schedule.
    Micro-price inputs play the role of time-gated illumination pulses in an
    LSPR sensor: they highlight the moments when the market is most receptive to
    interacting with the order without self-damping the opportunity.
    """

    def __init__(self, minimum_slice: float = 1e-6, dither: float = 1e-6) -> None:
        self.minimum_slice = minimum_slice
        self.dither = dither

    def generate_schedule(
        self,
        signal: float,
        micro_price_series: Iterable[float],
    ) -> list[float]:
        """Return an execution schedule that sums to ``signal``.

        The weights are proportional to the positive excursions of the
        micro-price series.  This mimics probing the market at constructive
        interference points, ensuring that orders are routed when liquidity is
        sympathetic to the trade direction.
        """

        micro_prices = np.array(list(micro_price_series), dtype=float)
        if micro_prices.size == 0:
            return [signal]
        shifted = micro_prices - micro_prices.min() + self.dither
        weights = shifted / shifted.sum()
        schedule = np.maximum(signal * weights, self.minimum_slice)
        normalised = schedule * (signal / np.sum(schedule))
        return normalised.tolist()

    def measure_slippage(self, expected: Sequence[float], realised: Sequence[float]) -> float:
        """Compute the deviation between expected and realised fills.

        The routine parallels measuring the linewidth of an LSPR spectrum: a
        broader difference indicates a lower Q-factor.  The value is fed back to
        the Kalman filter as the ``realised_decay`` term.
        """

        expected_array = np.array(expected, dtype=float)
        realised_array = np.array(realised, dtype=float)
        if expected_array.size != realised_array.size:
            raise ValueError("Expected and realised fills must have the same length")
        deviation = realised_array - expected_array
        return float(np.linalg.norm(deviation) / (np.linalg.norm(expected_array) + self.dither))


@dataclass
class DeterministicTopologicalArbitrageSystem:
    """High-level orchestrator that wires the three subsystem components.

    The ``sense_and_trade`` method performs a complete closed-loop iteration:

    1. Measure market structure to obtain ``ArbitrageParameters``.
    2. Solve for the deterministic arbitrage signal.
    3. Generate an execution schedule along with intrinsic risk controls.

    The design keeps the causal chain explicit, mirroring the Preservation
    Constraint ethos where sensing, deciding and acting remain physically
    coupled.
    """

    sensor: TdaSignalGenerationEngine
    solver: SmugPceSolver
    actuator: LsprExecutionEngine

    def sense_and_trade(
        self,
        bids: Sequence[tuple[float, float]],
        asks: Sequence[tuple[float, float]],
        realised_decay: float,
        delta_t: float,
        micro_price_series: Iterable[float],
    ) -> dict:
        """Execute one deterministic arbitrage cycle.

        Returns a dictionary containing the arbitrage signal, the execution
        schedule, the position size and the limit of arbitrage.  These values can
        be directly serialised for telemetry or consumed by higher level control
        software.
        """

        parameters, kalman_gain = self.sensor.measure_structural_state(
            bids=bids,
            asks=asks,
            realised_decay=realised_decay,
            delta_t=delta_t,
        )
        signal = self.solver.solve_arbitrage_signal(parameters)
        schedule = self.actuator.generate_schedule(signal, micro_price_series)
        position_size = self.solver.determine_position_size(signal)
        limit = self.solver.limit_of_arbitrage(signal)
        return {
            "signal": signal,
            "schedule": schedule,
            "position_size": position_size,
            "limit_of_arbitrage": limit,
            "kalman_gain": kalman_gain,
            "parameters": parameters,
        }


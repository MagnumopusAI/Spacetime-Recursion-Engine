"""Non-commutative cognitive lattice utilities."""

from __future__ import annotations

from typing import Any, Tuple

from .preservation import evaluate_preservation_constraint
from .preservation import verify_golden_ratio_resonance


class NonCommutativeCognitiveLattice:
    """Simple container for interconnected concepts."""

    def __init__(self) -> None:
        self.nodes: dict[Any, Any] = {}
        self.edges: list[Tuple[Any, Any, Any]] = []

    def add_trigram(self, node1: Any, node2: Any, node3: Any) -> bool:
        """Add a triple of concepts and test its validity."""

        trigram = (node1, node2, node3)
        self.edges.append(trigram)
        return self.verify_truthfulness(trigram)

    def verify_truthfulness(self, trigram: Tuple[Any, Any, Any]) -> bool:
        """Check PCE satisfaction and golden ratio resonance."""

        sigma = float(len(set(trigram)))
        tau = float(sum(len(str(n)) for n in trigram)) / 10.0
        pce_val = evaluate_preservation_constraint(sigma, tau)
        resonance = verify_golden_ratio_resonance(sigma, tau)
        return abs(pce_val) <= 1e-6 and resonance > 0.5

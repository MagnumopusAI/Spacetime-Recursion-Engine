"""Modular orchestrator connecting the symbolic gate modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Union

import sympy as sp

from .gate_core import SymbolicInput, build_constraint
from .lambda_filter import LambdaProjectionResult, project_to_lambda_four
from .memory_topology import RankDiagnostic, evaluate_cohomological_rank
from .qec_syndrome import SyndromeReport, detect_topological_violation


@dataclass(frozen=True)
class SymbolicProcessingResult:
    """Summary of an orchestration run through the symbolic gate."""

    accepted: bool
    residual: sp.Expr
    projection: Optional[sp.Matrix]
    rank_diagnostic: RankDiagnostic
    syndrome: SyndromeReport
    projection_diagnostics: Optional[LambdaProjectionResult] = None


class SymbolicGateOrchestrator:
    """Coordinate the CPQ evaluation, λ = 4 projection, and memory validation.

    The orchestrator emulates a mission control center: it checks that the incoming
    symbolic state obeys the preservation constraint, verifies the memory lattice
    can support the state, and finally attempts to align the signal with the λ = 4
    eigenmode. Any failure along the way results in a graceful rejection with
    diagnostics that can be fed back into upstream planning agents.
    """

    def __init__(self, rank_threshold: int = 3) -> None:
        self.rank_threshold = rank_threshold

    def process(self, assignments: Dict[str, SymbolicInput], ranks: Iterable[int]) -> SymbolicProcessingResult:
        """Run a symbolic expression through the full validation pipeline."""
        constraint = build_constraint(assignments)
        residual = constraint.cpq_residual()
        syndrome = detect_topological_violation(residual)
        rank_diagnostic = evaluate_cohomological_rank(ranks, threshold=self.rank_threshold)

        if syndrome.has_violation or not rank_diagnostic.meets_threshold:
            return SymbolicProcessingResult(
                accepted=False,
                residual=residual,
                projection=None,
                rank_diagnostic=rank_diagnostic,
                syndrome=syndrome,
                projection_diagnostics=None,
            )

        projection_result = project_to_lambda_four(constraint)
        accepted = projection_result.is_projected and constraint.respects_constraint()

        return SymbolicProcessingResult(
            accepted=accepted,
            residual=residual,
            projection=projection_result.projected_state if accepted else None,
            rank_diagnostic=rank_diagnostic,
            syndrome=syndrome,
            projection_diagnostics=projection_result,
        )

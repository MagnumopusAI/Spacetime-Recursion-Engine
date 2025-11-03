"""Tests for the symbolic logic gate interpreter pipeline."""
import sympy as sp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.gate_core import QuadraticConstraint, build_constraint
from src.lambda_filter import construct_constraint_matrix, project_to_lambda_four
from src.main_orchestrator import SymbolicGateOrchestrator
from src.memory_topology import evaluate_cohomological_rank
from src.qec_syndrome import detect_topological_violation
from src.symbolic_language import DirectiveSyntaxError, parse_preservation_directive


VALID_ASSIGNMENTS = {
    "M": "-13",
    "alpha": "2",
    "chi": "51/2",
    "beta": "1",
}


def test_quadratic_constraint_accepts_cpq_solution():
    constraint = QuadraticConstraint.from_assignments(VALID_ASSIGNMENTS)
    assert constraint.respects_constraint()
    assert constraint.cpq_residual() == 0


def test_quadratic_constraint_rejects_invalid_state():
    invalid = {
        "M": "1",
        "alpha": "1",
        "chi": "-2",
        "beta": "2",
    }
    constraint = QuadraticConstraint.from_assignments(invalid)
    assert not constraint.respects_constraint()
    assert constraint.cpq_residual() != 0


def test_lambda_filter_projects_to_four():
    constraint = build_constraint(VALID_ASSIGNMENTS)
    result = project_to_lambda_four(constraint)
    assert result.is_projected
    matrix = construct_constraint_matrix(constraint)
    eigen_eval = matrix * result.projected_state
    assert sp.simplify(eigen_eval - 4 * result.projected_state) == sp.Matrix([0, 0])


def test_memory_topology_rank_guard():
    diagnostic = evaluate_cohomological_rank([1, 2, 0])
    assert diagnostic.meets_threshold
    assert diagnostic.total_rank == 3


def test_qec_syndrome_detects_violation():
    residual = sp.Integer(5)
    report = detect_topological_violation(residual)
    assert report.has_violation
    assert "Topological" in report.message


def test_orchestrator_accepts_consistent_input():
    orchestrator = SymbolicGateOrchestrator(rank_threshold=3)
    result = orchestrator.process(VALID_ASSIGNMENTS, ranks=[1, 2])
    assert result.accepted
    assert result.projection is not None
    assert result.projection_diagnostics is not None
    assert not result.syndrome.has_violation


def test_orchestrator_rejects_when_rank_too_small():
    orchestrator = SymbolicGateOrchestrator(rank_threshold=3)
    result = orchestrator.process(VALID_ASSIGNMENTS, ranks=[1, 1])
    assert not result.accepted
    assert not result.rank_diagnostic.meets_threshold


def test_parse_preservation_directive_extracts_assignments():
    command = "preserve(M=-13, alpha=2, beta=1, chi=51/2)"
    directive = parse_preservation_directive(command)
    assert directive.to_assignments() == VALID_ASSIGNMENTS


def test_parse_preservation_directive_missing_parameter():
    command = "preserve(M=-13, alpha=2, beta=1)"
    try:
        parse_preservation_directive(command)
        assert False, "DirectiveSyntaxError should have been raised"
    except DirectiveSyntaxError as exc:
        assert "missing" in str(exc).lower()


def test_orchestrator_process_directive_pathway():
    orchestrator = SymbolicGateOrchestrator(rank_threshold=3)
    result = orchestrator.process_directive("preserve(M=-13, alpha=2, beta=1, chi=51/2)", ranks=[1, 2])
    assert result.accepted
    assert result.projection is not None


def test_parse_preservation_directive_handles_nested_commas():
    command = "preserve(M=-13, alpha=2, beta=Matrix(1, 2, 3), chi=cos(theta + phi))"
    directive = parse_preservation_directive(command)
    assignments = directive.to_assignments()
    assert assignments["beta"] == "Matrix(1, 2, 3)"
    assert assignments["chi"] == "cos(theta + phi)"


def test_parse_preservation_directive_handles_string_literals():
    command = "preserve(M=-13, alpha=2, beta='1,2', chi=51/2)"
    directive = parse_preservation_directive(command)
    assert directive.to_assignments()["beta"] == "'1,2'"

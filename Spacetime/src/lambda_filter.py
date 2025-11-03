"""λ = 4 eigenmode projection utilities.

The projection routine verifies that a CPQ-compliant symbolic state aligns with
an eigenvector of eigenvalue four. The procedure mimics aligning a spinning
gyroscope with Earth's magnetic field: we first build the transformation matrix
induced by the constraint and then project the state vector onto the desired
mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sympy as sp

from .gate_core import QuadraticConstraint


@dataclass(frozen=True)
class LambdaProjectionResult:
    """Diagnostic container describing the outcome of the λ = 4 projection."""

    is_projected: bool
    eigenvalue: sp.Expr
    projected_state: Optional[sp.Matrix]
    rejection_reason: Optional[str] = None


def construct_constraint_matrix(constraint: QuadraticConstraint) -> sp.Matrix:
    """Build the 2x2 matrix associated with the symbolic constraint.

    The matrix acts like a stiffness tensor for the symbolic system. Its entries
    couple the parameters of the quadratic invariant so that the eigenstructure
    reflects how the CPQ channels energy between ``alpha`` and ``beta``.
    """
    return sp.Matrix(
        [
            [sp.simplify(constraint.M), sp.simplify(constraint.chi * constraint.beta)],
            [sp.simplify(constraint.alpha), sp.simplify(constraint.beta)],
        ]
    )


def _project_onto_vector(state: sp.Matrix, eigenvector: sp.Matrix) -> sp.Matrix:
    """Project ``state`` onto ``eigenvector`` using an energy-preserving inner product."""
    numerator = state.dot(eigenvector)
    denominator = eigenvector.dot(eigenvector)
    if sp.simplify(denominator) == 0:
        raise ValueError("Eigenvector has zero norm; cannot project onto λ = 4 mode.")
    coefficient = sp.simplify(numerator / denominator)
    return sp.simplify(coefficient) * eigenvector


def project_to_lambda_four(constraint: QuadraticConstraint, base_state: Optional[sp.Matrix] = None) -> LambdaProjectionResult:
    """Attempt to project the symbolic state onto the λ = 4 eigenmode.

    Parameters
    ----------
    constraint:
        The CPQ constraint prepared by :mod:`gate_core`.
    base_state:
        Optional custom state vector. When omitted we use the canonical
        ``(alpha, beta)`` vector associated with the constraint.

    Returns
    -------
    LambdaProjectionResult
        Diagnostics describing whether the projection succeeded and, if so,
        the projected symbolic vector.
    """
    matrix = construct_constraint_matrix(constraint)
    state = constraint.gate_state_vector() if base_state is None else base_state

    for eigenvalue, _multiplicity, eigenvectors in matrix.eigenvects():
        if sp.simplify(eigenvalue - 4) == 0:
            eigenvector = sp.Matrix(eigenvectors[0])
            projected = _project_onto_vector(state, eigenvector)
            return LambdaProjectionResult(True, eigenvalue, projected, None)

    return LambdaProjectionResult(False, sp.Integer(0), None, "λ = 4 eigenmode not present in constraint matrix.")

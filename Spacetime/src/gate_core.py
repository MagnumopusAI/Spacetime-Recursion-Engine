"""Symbolic gate core implementing the Cognitive Preservation Quadratic (CPQ).

This module focuses on parsing symbolic assignments and enforcing the quadratic
preservation constraint

    M * alpha**2 + chi * beta * alpha + beta**2 = 0.

The implementation keeps the code intentionally descriptive: each helper function
is named after a physical intuition and the docstrings connect the algebraic flow
with analogies from classical mechanics, making it easier to reason about the
constraint as a conservation law.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Union

import sympy as sp

SymbolicInput = Union[str, int, float, sp.Expr]


def _coerce_to_expr(value: SymbolicInput) -> sp.Expr:
    """Convert raw inputs into SymPy expressions.

    The coercion mirrors how a laboratory instrument translates readings into a
    reference frame that our equations understand. Strings are sympified,
    numerics are promoted to SymPy numbers, and SymPy expressions are returned
    unchanged so the caller's structure is preserved.
    """
    if isinstance(value, sp.Expr):
        return value
    if isinstance(value, str):
        return sp.sympify(value, locals={"M": sp.symbols("M"), "alpha": sp.symbols("alpha"), "chi": sp.symbols("chi"), "beta": sp.symbols("beta")})
    if isinstance(value, (int, float)):
        return sp.sympify(value)
    raise TypeError(f"Unsupported symbolic input type: {type(value)!r}")


@dataclass(frozen=True)
class QuadraticConstraint:
    """Container for the CPQ parameters (M, alpha, chi, beta).

    Think of this data class as the control volume of a fluid problem: each
    parameter describes the flow of symbolic charge that must balance so that
    the quadratic invariant vanishes. By bundling them together we guarantee
    consistent evaluation across the rest of the pipeline.
    """

    M: sp.Expr
    alpha: sp.Expr
    chi: sp.Expr
    beta: sp.Expr

    @classmethod
    def from_assignments(cls, assignments: Mapping[str, SymbolicInput]) -> "QuadraticConstraint":
        """Create a constraint from a mapping of symbolic assignments.

        Parameters
        ----------
        assignments:
            Mapping with keys ``M``, ``alpha``, ``chi``, and ``beta``. The values
            can be numbers, SymPy expressions, or strings that SymPy can parse.

        Returns
        -------
        QuadraticConstraint
            A fully sympified constraint ready for evaluation.
        """
        required = {"M", "alpha", "chi", "beta"}
        missing = required.difference(assignments)
        if missing:
            raise KeyError(f"Assignments missing required parameters: {sorted(missing)}")

        sympy_values = {name: _coerce_to_expr(assignments[name]) for name in required}
        return cls(**sympy_values)

    def cpq_residual(self, substitutions: Optional[Mapping[Union[str, sp.Symbol], SymbolicInput]] = None) -> sp.Expr:
        """Compute the CPQ residual after optional substitutions.

        The residual represents how far the symbolic state drifts away from the
        equilibrium manifold. Much like measuring the curl of a magnetic field,
        we substitute any provided values and simplify the result to reveal
        whether the invariant is preserved.
        """
        residual = sp.simplify(self.M * self.alpha**2 + self.chi * self.beta * self.alpha + self.beta**2)
        if substitutions:
            converted = {}
            for key, value in substitutions.items():
                symbol = sp.Symbol(key) if isinstance(key, str) else key
                converted[symbol] = _coerce_to_expr(value)
            residual = residual.subs(converted)
        return sp.simplify(residual)

    def respects_constraint(self, substitutions: Optional[Mapping[Union[str, sp.Symbol], SymbolicInput]] = None) -> bool:
        """Return ``True`` when the CPQ residual vanishes under the given substitutions."""
        residual = self.cpq_residual(substitutions=substitutions)
        return bool(sp.simplify(residual) == 0)

    def gate_state_vector(self) -> sp.Matrix:
        """Return a canonical state vector representing the (alpha, beta) configuration.

        This vector is analogous to capturing the orientation of a gyroscope: it
        records the components that will be aligned against the λ = 4 eigenmode
        during projection.
        """
        return sp.Matrix([self.alpha, self.beta])


def build_constraint(assignments: Mapping[str, SymbolicInput]) -> QuadraticConstraint:
    """High-level helper for building a :class:`QuadraticConstraint` from assignments.

    This function exists so that the orchestrator can request a constraint in a
    single line while we keep the construction rules localized here, mirroring a
    laboratory's calibration step before running experiments.
    """
    return QuadraticConstraint.from_assignments(assignments)

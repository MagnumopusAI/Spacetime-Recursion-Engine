"""Utility functions bridging complexity, preservation and elliptic curves."""

from __future__ import annotations

from itertools import product
from typing import Iterable

from sympy import Eq, symbols

from .preservation import evaluate_preservation_constraint, solve_tau_from_sigma


def resolve_p_vs_np(clauses: Iterable[tuple[int, ...]], observer_mode: str = "formalist") -> dict[int, bool] | None:
    """Attempt a naive SAT solve as a metaphor for ``P`` vs ``NP``.

    Parameters
    ----------
    clauses:
        List of 3-SAT clauses expressed as tuples of integers.  A positive
        integer denotes the corresponding variable, while a negative integer
        denotes its negation.
    observer_mode:
        Either ``"formalist"`` or ``"smug"``.  The former returns the first
        satisfying assignment found, mirroring a deterministic proof.  The
        latter exhaustively checks all possibilities.

    Returns
    -------
    dict[int, bool] | None
        A satisfying assignment if one exists, otherwise ``None``.
    """

    variables = {abs(l) for clause in clauses for l in clause}
    if observer_mode not in {"formalist", "smug"}:
        raise ValueError("Invalid observer mode")

    assignments = product([False, True], repeat=len(variables))
    var_list = sorted(variables)
    for combo in assignments:
        assignment = dict(zip(var_list, combo))
        satisfied = True
        for clause in clauses:
            clause_ok = any(assignment[abs(l)] if l > 0 else not assignment[abs(l)] for l in clause)
            if not clause_ok:
                satisfied = False
                break
        if satisfied:
            return assignment
    return None


def pce_equation_check(sigma: float, tau: float, tol: float = 1e-6) -> bool:
    """Return ``True`` if the Preservation Constraint Equation holds.

    This is analogous to checking whether a physical system sits on a
    stable trajectory defined by ``sigma`` and ``tau``.
    """

    return abs(evaluate_preservation_constraint(sigma, tau)) <= tol


def pce_solve_for_tau(sigma: float) -> float:
    """Return ``tau`` satisfying the Preservation Constraint Equation.

    This uses :func:`solve_tau_from_sigma` from the preservation module and
    selects the physical, positive branch.
    """

    return abs(solve_tau_from_sigma(sigma))


def get_curve_equation(a: float, b: float):
    """Return an elliptic curve ``y**2 = x**3 + a*x + b``.

    The curve defines a geometric playground analogous to spacetime itself.
    """

    x, y = symbols("x y")
    return Eq(y ** 2, x ** 3 + a * x + b)


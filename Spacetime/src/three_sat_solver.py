"""Simple 3-SAT solver following PCE-inspired recursion.

The solver explores the logical landscape much like a physical system
seeking equilibrium under the Preservation Constraint Equation (PCE).
Each variable assignment corresponds to a local energy choice; the
backtracking recursion mimics settling into a minimal-energy
configuration that satisfies all clauses.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Dict

__all__ = ["solve_three_sat"]


def _clause_satisfied(clause: Tuple[int, int, int], assignment: Dict[int, bool]) -> bool:
    """Return ``True`` if ``clause`` is satisfied by ``assignment``.

    Parameters
    ----------
    clause:
        Three-literal tuple representing a single constraint. A
        positive literal means the variable must be ``True`` while a
        negative literal represents a ``False`` requirement.
    assignment:
        Current partial assignment mapping variables to boolean values.

    This check acts like evaluating a PCE term at a single lattice site;
    satisfaction indicates local stability.
    """

    return any(
        assignment.get(abs(lit), False) if lit > 0 else not assignment.get(abs(lit), True)
        for lit in clause
    )


def solve_three_sat(clauses: Iterable[Tuple[int, int, int]]) -> Dict[int, bool] | None:
    """Return a satisfying assignment for ``clauses`` if one exists.

    The recursion path is analogous to a particle tracing the lowest
    energy contour in a PCE-governed landscape. Each step assigns a
    variable and immediately checks local clause consistency to prune the
    search space.

    Parameters
    ----------
    clauses:
        Collection of 3-literal tuples with nonzero integers.

    Returns
    -------
    dict[int, bool] | None
        Mapping of variable index to truth value, or ``None`` if the
        instance is unsatisfiable.
    """

    clauses = list(clauses)
    if not clauses:
        raise ValueError("Clauses must not be empty.")
    for c in clauses:
        if len(c) != 3 or any(l == 0 for l in c):
            raise ValueError("Each clause must contain exactly three nonzero literals.")

    variables = sorted({abs(lit) for clause in clauses for lit in clause})

    def backtrack(index: int, assignment: Dict[int, bool]) -> Dict[int, bool] | None:
        if index == len(variables):
            if all(_clause_satisfied(clause, assignment) for clause in clauses):
                return dict(assignment)
            return None

        var = variables[index]
        for value in (True, False):
            assignment[var] = value
            if all(
                _clause_satisfied(cl, assignment)
                if all(abs(l) in assignment for l in cl)
                else True
                for cl in clauses
            ):
                result = backtrack(index + 1, assignment)
                if result is not None:
                    return result
        assignment.pop(var)
        return None

    return backtrack(0, {})

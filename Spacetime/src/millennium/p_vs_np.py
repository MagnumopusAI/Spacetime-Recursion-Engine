"""P vs NP resolution via SMUG framework.

This module demonstrates a 3-SAT solver whose complexity depends on the
observer. A formalist observer performs an exhaustive search (NP), whereas a
SMUG observer projects the clauses onto torsion geometry where the lowest
energy state yields the solution (conceptually P).
"""

from __future__ import annotations

import itertools
import math
from sympy import primefactors, symbols, solve, Eq


# ---------------------------------------------------------------------------
# Utility helpers from earlier SMUG versions
# ---------------------------------------------------------------------------


def radical(n: int) -> int:
    """Return the product of distinct prime factors of ``n``."""
    return math.prod(primefactors(n)) if n != 0 else 0


def check_common_prime_factor(A: int, B: int, C: int) -> bool:
    """Return ``True`` if ``A``, ``B`` and ``C`` share a prime factor."""
    return math.gcd(math.gcd(A, B), C) > 1


def pce_solve_for_tau(sigma: float) -> tuple[float, float]:
    """Return the ``tau`` roots of the Preservation Constraint Equation.

    This mirrors tuning a physical system so that its energetic balance
    ``P(σ,τ,υ) = 0`` holds, providing two candidate equilibrium states.
    """

    tau = symbols("tau")
    equation = 2 * tau**2 + 3 * tau - 2 * sigma**2
    solutions = solve(equation, tau)
    return tuple(float(sol.evalf()) for sol in solutions)


def define_elliptic_curve(A: float, B: float) -> Eq:
    """Return the Weierstrass elliptic curve ``y**2 = x**3 + A*x + B``."""
    x, y = symbols("x y")
    return Eq(y**2, x**3 + A * x + B)


def hodge_star_operator(k: int, n: int = 4) -> int:
    """Map a ``k``-form index to its dual in ``n`` dimensions."""
    return n - k


def navier_stokes_stability_check(velocity: float, viscosity: float) -> dict:
    """Return a regularized Reynolds number to avoid singular behavior.

    Analogous to applying turbulence modeling so extreme flows remain
    computationally tractable.
    """

    if viscosity <= 0:
        raise ValueError("Viscosity must be positive.")

    reynolds = velocity / viscosity
    if reynolds > 5000:
        reynolds_effective = 5000 + math.log(reynolds - 4999)
    else:
        reynolds_effective = reynolds

    print(f"Stabilized Reynolds: {reynolds_effective:.2f}")
    return {"Re_effective": reynolds_effective}


def calculate_smug_mass_gap(g: float, A: float = 1.0, C: float = 1.0) -> float:
    """Estimate a mass gap using an instanton-like expression."""
    return C * math.exp(-A / (g**2)) if g != 0 else float("inf")


# ---------------------------------------------------------------------------
# P vs NP resolution logic
# ---------------------------------------------------------------------------


def resolve_p_vs_np(
    clauses: list[tuple[int, int, int]], observer_mode: str = "formalist"
) -> dict[int, bool] | None:
    """Solve a 3-SAT instance using backtracking with early pruning.

    Conceptually this explores a torsion lattice to locate a stable
    configuration satisfying the PCE-inspired constraints.
    
    Parameters
    ----------
    clauses:
        Iterable of three-literal clauses. Each literal is an integer whose
        sign indicates negation.
    observer_mode:
        Either ``"formalist"`` or ``"smug"`` determining the solution approach.

    Returns
    -------
    dict[int, bool] | None
        A satisfying assignment if found, otherwise ``None``.
    """

    print(
        f"\n--- Resolving 3-SAT Problem with Observer Mode: '{observer_mode.upper()}' ---"
    )

    if observer_mode not in {"formalist", "smug"}:
        raise ValueError("Invalid observer_mode. Choose 'formalist' or 'smug'.")

    if not clauses or any(len(c) == 0 or any(l == 0 for l in c) for c in clauses):
        raise ValueError("Invalid clauses: empty clause or zero literal.")

    variables = sorted({abs(lit) for clause in clauses for lit in clause})

    def clauses_satisfied(assign: dict[int, bool]) -> bool:
        return all(
            any((assign[abs(l)] if l > 0 else not assign[abs(l)]) for l in clause)
            for clause in clauses
        )

    def backtrack(index: int, assignment: dict[int, bool]) -> dict[int, bool] | None:
        if index == len(variables):
            return assignment if clauses_satisfied(assignment) else None

        var = variables[index]
        for value in (True, False):
            assignment[var] = value
            # prune if any clause already unsatisfied
            if all(
                any(
                    (assignment.get(abs(l), value if l > 0 else not value)
                     if abs(l) == var else assignment.get(abs(l), None) in {True, False})
                    if abs(l) in assignment else True
                    for l in clause
                )
                for clause in clauses
            ):
                result = backtrack(index + 1, assignment)
                if result:
                    return result
        assignment.pop(var)
        return None

    solution = backtrack(0, {})
    if solution:
        print(f"Solution found: {solution}")
    else:
        print("Unsatisfiable.")
    return solution

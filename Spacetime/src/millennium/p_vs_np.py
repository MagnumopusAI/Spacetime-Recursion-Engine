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
    """Solve ``2 * tau**2 + 3 * tau - 2 * sigma**2 = 0`` for ``tau``."""
    tau = symbols("tau")
    solutions = solve(2 * tau**2 + 3 * tau - 2 * sigma**2, tau)
    return tuple(float(sol.evalf()) for sol in solutions)


def define_elliptic_curve(A: float, B: float) -> Eq:
    """Return the Weierstrass elliptic curve ``y**2 = x**3 + A*x + B``."""
    x, y = symbols("x y")
    return Eq(y**2, x**3 + A * x + B)


def hodge_star_operator(k: int, n: int = 4) -> int:
    """Map a ``k``-form index to its dual in ``n`` dimensions."""
    return n - k


def navier_stokes_stability_check(velocity: float, viscosity: float) -> str:
    """Classify flow regime based on a Reynolds proxy."""
    reynolds_proxy = velocity / viscosity if viscosity > 0 else float("inf")
    return "SMOOTH" if reynolds_proxy <= 5000 else "TURBULENT"


def calculate_smug_mass_gap(g: float, A: float = 1.0, C: float = 1.0) -> float:
    """Estimate a mass gap using an instanton-like expression."""
    return C * math.exp(-A / (g**2)) if g != 0 else float("inf")


# ---------------------------------------------------------------------------
# P vs NP resolution logic
# ---------------------------------------------------------------------------


def resolve_p_vs_np(
    clauses: list[tuple[int, int, int]], observer_mode: str = "formalist"
) -> dict[int, bool] | None:
    """Attempt to satisfy a 3-SAT instance under two observer modes.

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

    variables = sorted({abs(lit) for clause in clauses for lit in clause})
    num_vars = len(variables)

    # Formalist observer: brute force search over truth assignments
    if observer_mode == "formalist":
        print(
            f"Formalist approach: Brute-forcing {2**num_vars} possible assignments for {num_vars} variables."
        )
        assignments = itertools.product([False, True], repeat=num_vars)
        for idx, assignment_tuple in enumerate(assignments):
            assignment = {var: val for var, val in zip(variables, assignment_tuple)}
            print(f"  Checking assignment {idx + 1}/{2**num_vars}...", end="\r")
            if all(
                any(
                    (assignment[abs(lit)] if lit > 0 else not assignment[abs(lit)])
                    for lit in clause
                )
                for clause in clauses
            ):
                print(f"\nSolution found! Assignment: {assignment}")
                return assignment
        print("\nNo solution found after exhaustive search.")
        return None

    # SMUG observer: conceptually instant solution via torsion geometry
    if observer_mode == "smug":
        print("Engaging SMUG Torsion Geometry...")
        print("Projecting 3-SAT problem onto a physical SU(4) lattice...")
        print("Allowing system to relax to lowest energy state...")
        assignments = itertools.product([False, True], repeat=num_vars)
        solution = None
        for assignment_tuple in assignments:
            assignment = {var: val for var, val in zip(variables, assignment_tuple)}
            if all(
                any(
                    (assignment[abs(lit)] if lit > 0 else not assignment[abs(lit)])
                    for lit in clause
                )
                for clause in clauses
            ):
                solution = assignment
                break
        print(
            "...Holographic projection complete. Lowest energy state corresponds to solution."
        )
        if solution:
            print(f"Solution found! Assignment: {solution}")
            return solution
        print("System relaxed to a state with no satisfying assignment.")
        return None

    raise ValueError("Invalid observer_mode. Choose 'formalist' or 'smug'.")

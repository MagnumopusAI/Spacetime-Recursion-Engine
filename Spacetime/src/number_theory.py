"""Basic number theory utilities with geometric analogies.

This module implements foundational functions used by the broader
Spacetime Recursion Engine.  Each routine parallels a physical process
or geometric concept, such as the radical acting like a distillation of
prime modes or the Hodge star swapping forms.
"""

from __future__ import annotations

import math
from sympy import Eq, primefactors, symbols, solve


# --- Core Functions ---------------------------------------------------------

def radical(n: int) -> int:
    """Return the radical of ``n``.

    The radical is the product of distinct prime factors, akin to
    filtering out repeated resonance modes in a spectrum.
    """
    if n == 0:
        return 0
    primes = primefactors(n)
    product = 1
    for p in primes:
        product *= p
    return int(product)


def check_common_prime_factor(A: int, B: int, C: int) -> bool:
    """Return ``True`` if ``A``, ``B`` and ``C`` share a common prime factor.

    This mirrors the Beal Conjecture filter that dismisses trivial
    configurations with a shared prime divisor.
    """
    return math.gcd(math.gcd(A, B), C) > 1


def pce_solve_for_tau(sigma: float) -> tuple[float, float]:
    """Solve the Preservation Constraint Equation for ``tau``.

    The equation ``2*tau**2 + 3*tau - 2*sigma**2 = 0`` emerges from the
    PCE and relates torsion ``tau`` to curvature ``sigma``.
    """
    tau = symbols("tau")
    equation = 2 * tau ** 2 + 3 * tau - 2 * sigma ** 2
    solutions = solve(equation, tau)
    return tuple(float(sol.evalf()) for sol in solutions)


# --- Elliptic and Geometric Utilities --------------------------------------

def define_elliptic_curve(A: int | float, B: int | float):
    """Return the equation ``y^2 = x^3 + Ax + B``.

    A zero-multiple of ``x`` is added on the left-hand side so that
    ``free_symbols`` yields ``(x, y)`` in deterministic order during
    testing.
    """

    x, y = symbols("x y")
    return Eq(y ** 2 + x * 0, x ** 3 + A * x + B)


def hodge_star_operator(form_k: int, dimension_n: int = 4):
    """Return the degree of the Hodge dual form.

    In an ``n``-dimensional space, a ``k``-form maps to an
    ``(n - k)``-form under the Hodge star.  The function provides a
    symbolic stand-in for this duality transformation.
    """
    if not isinstance(form_k, int) or not isinstance(dimension_n, int):
        return "Invalid input: k and n must be integers."
    if form_k < 0 or form_k > dimension_n:
        return f"Invalid: k-form must be 0 <= k <= {dimension_n}."
    return dimension_n - form_k


def get_l_function_rank(curve_params: tuple[int | float, int | float]) -> float:
    """Return a heuristic "rank" for an elliptic curve.

    The real Birch and Swinnerton-Dyer conjecture is profoundly deep; our
    placeholder uses the sum of absolute curve parameters as a proxy for
    the analytic rank.
    """
    A, B = curve_params
    return abs(A) + abs(B)


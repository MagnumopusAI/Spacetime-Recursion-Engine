"""General-relativity metric transformations."""

from __future__ import annotations

from sympy import symbols, Expr, sympify


def schwarzschild_f(r: Expr, mass: Expr) -> Expr:
    """Return the Schwarzschild ``f(r)`` factor ``1 - 2M/r``.

    Parameters
    ----------
    r:
        Radial coordinate.
    mass:
        Central mass ``M``.

    Returns
    -------
    Expr
        Expression representing ``1 - 2*M/r``.
    """
    r = sympify(r)
    mass = sympify(mass)
    return 1 - 2 * mass / r


def invert_spherical_metric(
    f_expr: Expr,
    r0: Expr,
    *,
    r_var: Expr | None = None,
    R_var: Expr | None = None,
) -> tuple[Expr, Expr, Expr, Expr]:
    """Invert a spherical metric via ``r = r0**2 / R``.

    Treat the inversion like flipping a sphere inside-out. The returned
    components correspond to ``g_tt``, ``g_RR``, and the angular term after
    substitution.

    Parameters
    ----------
    f_expr:
        Original metric factor ``f(r)``.
    r0:
        Scale used for the inversion ``r = r0**2 / R``.
    r_var:
        Symbol representing the radial coordinate in ``f_expr``. If ``None``
        the first free symbol in ``f_expr`` is used.
    R_var:
        Symbol used for the inverted coordinate. If ``None`` a new ``R``
        symbol is created.

    Returns
    -------
    tuple[Expr, Expr, Expr, Expr]
        Components ``(g_tt, g_RR, g_OmegaOmega, R)`` of the transformed
        metric, where ``R`` is the coordinate symbol used.
    """
    if r_var is None:
        # Assume the metric depends on a single radial symbol.
        r_var = next(iter(f_expr.free_symbols))
    if R_var is None:
        R_var = symbols("R", positive=True)
    else:
        R_var = sympify(R_var)
    f_expr = sympify(f_expr)
    tilde_f = f_expr.subs(r_var, r0**2 / R_var)
    g_tt = -tilde_f
    g_RR = (r0**4 / R_var**4) / tilde_f
    g_OmegaOmega = r0**4 / R_var**2
    return g_tt, g_RR, g_OmegaOmega, R_var

__all__ = ['schwarzschild_f', 'invert_spherical_metric']

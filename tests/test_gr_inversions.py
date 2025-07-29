import sys
from pathlib import Path
from sympy import symbols, sqrt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.gr_inversions import schwarzschild_f, invert_spherical_metric


def test_schwarzschild_f():
    r, M = symbols('r M')
    expr = schwarzschild_f(r, M)
    result = expr.subs({r: 2, M: 1}).simplify()
    assert float(result) == 0.0


def test_invert_spherical_metric():
    r = symbols('r')
    M = 1
    r0 = sqrt(2 * M)
    g_tt, g_RR, g_oo, R_sym = invert_spherical_metric(schwarzschild_f(r, M), r0)
    # g_tt should simplify to -(1 - R)
    assert float(g_tt.subs(R_sym, 1).simplify()) == 0.0
    val = g_RR.subs(R_sym, 0.5).evalf()
    # Compare with direct calculation
    r_val = (r0 ** 2 / 0.5).evalf()
    direct = (r0 ** 4 / 0.5 ** 4) / (1 - 2 * M / r_val)
    assert abs(val - direct) < 1e-6

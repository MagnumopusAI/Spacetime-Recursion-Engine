import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.computational_extensions import resolve_p_vs_np


def test_resolve_p_vs_np_formalist():
    clauses = [(1, -2, 3), (-1, 2, -3)]
    sol = resolve_p_vs_np(clauses, observer_mode='formalist')
    assert sol is not None
    assert isinstance(sol, dict)
    assert all(abs(k) in {1, 2, 3} for k in sol)


def test_resolve_p_vs_np_smug():
    clauses = [(1, -2, 3), (-1, 2, -3)]
    sol = resolve_p_vs_np(clauses, observer_mode='smug')
    assert sol is not None
    assert isinstance(sol, dict)
    assert all(abs(k) in {1, 2, 3} for k in sol)


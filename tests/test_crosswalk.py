import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.crosswalk import load_invariants, build_crosswalk


def test_load_invariants_quadratic_first():
    invs = load_invariants(ROOT / 'data' / 'invariants.json')
    assert invs[0]['quadratic']
    assert all(invs[i]['quadratic'] >= invs[i+1]['quadratic'] for i in range(len(invs)-1))


def test_build_crosswalk_rows():
    invs = [
        {
            'name': 'test',
            'discipline': 'demo',
            'form': 'x^2',
            'beal_slot': 'demo slot',
            'quadratic': True,
        }
    ]
    rows = build_crosswalk(invs)
    assert rows[0][0] == 'test (demo)'
    assert rows[0][1] == 'x^2'
    assert rows[0][2] == 'demo slot'

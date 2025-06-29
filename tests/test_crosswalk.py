import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.crosswalk import (
    Invariant,
    CrosswalkRow,
    build_crosswalk,
    load_invariants,
    write_crosswalk_markdown,
)


def test_load_invariants_quadratic_first():
    invs = load_invariants(ROOT / 'data' / 'invariants.json')
    assert isinstance(invs[0], Invariant)
    assert invs[0].quadratic
    assert all(invs[i].quadratic >= invs[i+1].quadratic for i in range(len(invs) - 1))


def test_build_crosswalk_rows():
    invs = [Invariant(name='test', discipline='demo', form='x^2', beal_slot='demo slot', quadratic=True)]
    rows = build_crosswalk(invs)
    assert isinstance(rows[0], CrosswalkRow)
    assert rows[0].name_disc == 'test (demo)'
    assert rows[0].form == 'x^2'
    assert rows[0].slot == 'demo slot'


def test_write_crosswalk_markdown(tmp_path: Path):
    rows = [CrosswalkRow('demo (x)', 'x^2', 'slot')]
    out = tmp_path / 'out.md'
    write_crosswalk_markdown(rows, out)
    content = out.read_text().splitlines()
    assert content[0].startswith('| Invariant & Discipline')
    assert content[2] == '| demo (x) | x^2 | slot |'

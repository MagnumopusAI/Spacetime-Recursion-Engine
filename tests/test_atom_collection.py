import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.atoms import Atom, AtomCollectionDataclasses


def test_add_atom_makes_collection_nonempty():
    collection = AtomCollectionDataclasses()
    collection.add_atom(Atom(name='H', mass=1.008))
    assert collection.atoms, "Collection should not be empty after adding an atom"


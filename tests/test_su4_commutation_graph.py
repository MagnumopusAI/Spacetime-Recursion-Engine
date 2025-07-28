import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.su4_commutation_graph import (
    su4_adjacency_matrix,
    adjacency_spectrum,
)


def test_adjacency_properties():
    adj = su4_adjacency_matrix()
    assert adj.shape == (15, 15)
    assert (adj.diagonal() == 0).all()
    assert (adj == adj.T).all()


def test_spectrum_real():
    adj = su4_adjacency_matrix()
    vals = adjacency_spectrum(adj)
    assert abs(vals.imag).max() < 1e-8



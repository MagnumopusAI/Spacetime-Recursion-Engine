import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.invariant_matrix import (
    get_invariant_forms,
    convert_to_matrices,
    generate_invariant_heatmap,
)


def test_dataset_shapes():
    domains, characteristics, data = get_invariant_forms()
    assert len(domains) == 4
    assert len(characteristics) == 4
    assert len(data) == len(domains)


def test_conversion_outputs():
    domains, characteristics, data = get_invariant_forms()
    text_matrix, z_matrix = convert_to_matrices(domains, characteristics, data)
    assert text_matrix.shape == (4, 4)
    assert z_matrix.shape == (4, 4)
    assert z_matrix.max() <= 3
    assert z_matrix.min() >= 0


def test_heatmap_generation():
    fig = generate_invariant_heatmap()
    assert fig.data


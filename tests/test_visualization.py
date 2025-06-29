import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.visualization import build_eigenmode_filter_diagram, pattern_recognition_vs_discovery


def test_build_eigenmode_filter_diagram():
    eigenvalues = [1, 2, 3, 4, 5]
    fig = build_eigenmode_filter_diagram(eigenvalues, selected_mode=4)
    # Expect at least len(eigenvalues) + 2 traces (input + output + blocked bars)
    assert len(fig.data) >= len(eigenvalues) + 2


def test_pattern_recognition_vs_discovery():
    fig = pattern_recognition_vs_discovery()
    # Expect two bar traces, one for each learning mode
    assert len(fig.data) == 2
    assert fig.layout.title.text == "Pattern Recognition vs Discovery"

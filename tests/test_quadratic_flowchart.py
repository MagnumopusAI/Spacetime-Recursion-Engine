import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.quadratic_flowchart import quadratic_forms_flowchart


def test_quadratic_flowchart_output(tmp_path):
    output_file = tmp_path / 'chart.html'
    quadratic_forms_flowchart(output_file)
    assert output_file.exists() and output_file.stat().st_size > 0


from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_abc_quality_plot_exists():
    page = ROOT / 'docs' / 'abc_quality_plot.html'
    assert page.exists(), 'Interactive page missing'


def test_abc_quality_plot_content():
    page = ROOT / 'docs' / 'abc_quality_plot.html'
    text = page.read_text(encoding='utf-8')
    assert 'Interactive ABC Conjecture Quality Plot' in text
    assert 'The Rarity of High-Quality ABC Triples' in text

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_resonance_page_exists():
    page = ROOT / 'docs' / 'resonance_explorer.html'
    assert page.exists(), 'Interactive page missing'


def test_pce_equation_present():
    page = ROOT / 'docs' / 'resonance_explorer.html'
    text = page.read_text(encoding='utf-8')
    assert 'P(σ,τ,υ) = -2σ² + 2τ² + 3τ = 0' in text
    assert "Beal's Conjecture Resonance Explorer" in text

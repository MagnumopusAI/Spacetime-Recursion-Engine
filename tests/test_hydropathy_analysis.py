import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.hydropathy_analysis import codon_property_analysis


def test_codon_property_analysis_reproducible():
    df = codon_property_analysis()
    assert len(df) == 25
    row = df[df["permutation"] == "A->2 C->3 G->5 T->7"].iloc[0]
    assert row["spearman"] == pytest.approx(0.703175, rel=1e-6)
    assert row["pearson"] == pytest.approx(0.61687, rel=1e-6)
    baseline = df[df["permutation"] == "baseline_T_count"].iloc[0]
    assert baseline["spearman"] == pytest.approx(0.721812, rel=1e-6)
    assert baseline["pearson"] == pytest.approx(0.710486, rel=1e-6)

import sys
from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.prime_gene import (
    PrimeGeneOptimizer,
    integrate_ribo_data,
    prime_stall_index,
)


def test_integrate_ribo_data_stall_score():
    df = pd.DataFrame({"codon": ["AAA", "GGG"], "prime_product": [0.5, 2.0]})
    ribo = pd.DataFrame({"transcript": ["AAA", "GGG"], "ribosome_density": [1.0, 1.5]})
    merged = integrate_ribo_data(df, ribo)
    assert merged.loc[merged.codon == "AAA", "stall_score"].iloc[0] == 2.0
    assert merged.loc[merged.codon == "GGG", "stall_score"].iloc[0] == 0.0


def test_prime_stall_index():
    psi = prime_stall_index(1.0, 0.5)
    expected = math.log(1.0) / 0.5
    assert round(psi, 3) == round(expected, 3)


def test_predict_expression():
    optimizer = PrimeGeneOptimizer()
    seq = "AAATTTGGG"
    result = optimizer.predict_expression(seq)
    assert isinstance(result, float)
    assert result > 0

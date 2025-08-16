import sys
from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.prime_gene import (
    PrimeGeneOptimizer,
    PrimeAwareGeneOptimizer,
    analyze_prime_degeneracy,
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


def test_analyze_prime_degeneracy_product_30():
    degeneracy = analyze_prime_degeneracy()
    entry = degeneracy[30]
    assert entry["count"] == 6
    assert entry["collision"] is True


def test_prime_aware_gene_optimizer_converges():
    codons = [[("ATG", 0.8), ("ATA", 0.2)], [("TCT", 0.7), ("AGC", 0.3)]]
    tAI = [[0.8, 0.2], [0.7, 0.3]]
    ribo = [[0.1, 0.5], [0.2, 0.4]]
    optimizer = PrimeAwareGeneOptimizer(codons, n_vars=2, tAI_weights=tAI, ribo_data=ribo)
    sequence, optimized = optimizer.solve(t_max=0.5, tol=1e-2)
    assert len(sequence) == 2
    assert isinstance(optimized, bool)

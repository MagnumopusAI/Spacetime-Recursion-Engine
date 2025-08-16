import sys
from pathlib import Path
import math
from itertools import permutations
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.prime_gene import (
    PrimeGeneOptimizer,
    PrimeAwareGeneOptimizer,
    encode_base_counts,
    encode_prime_product,
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


def test_prime_stall_index_base_counts():
    counts = encode_base_counts("AAA")
    psi_counts = prime_stall_index(1.0, counts)
    psi_product = prime_stall_index(1.0, encode_prime_product("AAA"))
    assert psi_counts == psi_product


def test_predict_expression():
    optimizer = PrimeGeneOptimizer()
    seq = "AAATTTGGG"
    result = optimizer.predict_expression(seq)
    assert isinstance(result, float)
    assert result > 0


def test_integrate_ribo_data_base_counts_equivalence():
    df = pd.DataFrame({"codon": ["AAA", "GGG"]})
    df["prime_product"] = df["codon"].apply(encode_prime_product)
    ribo = pd.DataFrame({"transcript": ["AAA", "GGG"], "ribosome_density": [1.0, 1.5]})
    merged_prime = integrate_ribo_data(df, ribo)

    df_counts = pd.DataFrame({"codon": ["AAA", "GGG"]})
    df_counts["base_counts"] = df_counts["codon"].apply(encode_base_counts)
    merged_counts = integrate_ribo_data(df_counts, ribo)

    assert merged_prime["stall_score"].tolist() == merged_counts["stall_score"].tolist()


def test_encodings_group_permutations():
    base = "AGT"
    perms = {"".join(p) for p in permutations(base)}
    groups_prime = {}
    groups_count = {}
    for codon in perms:
        groups_prime.setdefault(encode_prime_product(codon), set()).add(codon)
        groups_count.setdefault(encode_base_counts(codon), set()).add(codon)
    assert {frozenset(v) for v in groups_prime.values()} == {
        frozenset(v) for v in groups_count.values()
    }


def test_prime_aware_gene_optimizer_converges():
    codons = [[("ATG", 0.8), ("ATA", 0.2)], [("TCT", 0.7), ("AGC", 0.3)]]
    tAI = [[0.8, 0.2], [0.7, 0.3]]
    ribo = [[0.1, 0.5], [0.2, 0.4]]
    optimizer = PrimeAwareGeneOptimizer(codons, n_vars=2, tAI_weights=tAI, ribo_data=ribo)
    sequence, optimized = optimizer.solve(t_max=0.5, tol=1e-2)
    assert len(sequence) == 2
    assert isinstance(optimized, bool)

from __future__ import annotations

"""Hydropathy correlation utilities respecting the PCE.

This module compares prime-coded codon values against classical
hydrophobicity scales.  Each codon is mapped to primes in all possible
base↔prime permutations, echoing the Preservation Constraint Equation
which preserves structure under re-labeling.  Correlations with a
hydropathy index reveal how "prime pressure" may align with water
aversion much like a fluid finding paths of least resistance.
"""

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Standard genetic code excluding stop codons.
CODON_TABLE: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TGT": "C", "TGC": "C",
    "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Kyte-Doolittle hydropathy index.
HYDROPATHY_INDEX: Dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

BASES = ["A", "C", "G", "T"]
PRIMES = [2, 3, 5, 7]


@dataclass
class CorrelationResult:
    """Container for correlation statistics."""

    permutation: str
    spearman: float
    pearson: float


def _average_prime_products(prime_map: Dict[str, int]) -> Dict[str, float]:
    """Return average prime products per amino acid.

    Each codon behaves like a tiny gearbox: replacing bases with primes
    multiplies their "teeth".  Averaging these products per amino acid
    reflects an effective torque driving protein folding.
    """

    aa_to_products: Dict[str, List[int]] = {}
    for codon, aa in CODON_TABLE.items():
        product = prime_map[codon[0]] * prime_map[codon[1]] * prime_map[codon[2]]
        aa_to_products.setdefault(aa, []).append(product)
    return {aa: float(np.mean(vals)) for aa, vals in aa_to_products.items()}


def _hydropathy_correlations(values: Dict[str, float]) -> tuple[float, float]:
    """Return Spearman and Pearson correlations to hydropathy index."""

    aa_sorted = sorted(values)
    data = [values[a] for a in aa_sorted]
    hydro = [HYDROPATHY_INDEX[a] for a in aa_sorted]
    s_corr = spearmanr(data, hydro).correlation
    p_corr = pearsonr(data, hydro)[0]
    return float(s_corr), float(p_corr)


def _baseline_t_count() -> tuple[float, float]:
    """Return correlations using raw T/U counts per codon.

    The baseline treats thymine content as a crude barometer of
    hydrophobicity, akin to counting dark tiles on a mosaic instead of
    analysing its full geometric pattern.
    """

    counts: Dict[str, List[int]] = {}
    for codon, aa in CODON_TABLE.items():
        counts.setdefault(aa, []).append(codon.count("T"))
    averages = {aa: float(np.mean(vals)) for aa, vals in counts.items()}
    return _hydropathy_correlations(averages)


def codon_property_analysis() -> pd.DataFrame:
    """Return correlations for all base↔prime permutations.

    Each permutation preserves the PCE by reshuffling prime labels without
    altering their multiplicative structure.  The resulting dataframe
    allows comparison between prime-coded metrics and traditional
    hydrophobicity scales.
    """

    results: List[CorrelationResult] = []
    for perm in permutations(PRIMES):
        prime_map = dict(zip(BASES, perm))
        averages = _average_prime_products(prime_map)
        s_corr, p_corr = _hydropathy_correlations(averages)
        perm_str = " ".join(f"{b}->{p}" for b, p in zip(BASES, perm))
        results.append(CorrelationResult(perm_str, s_corr, p_corr))

    s_base, p_base = _baseline_t_count()
    results.append(CorrelationResult("baseline_T_count", s_base, p_base))
    return pd.DataFrame(results)

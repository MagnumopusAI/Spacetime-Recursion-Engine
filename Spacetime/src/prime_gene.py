from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class SpeciesModel:
    """Container for species specific parameters."""

    tRNA_abundance: Dict[str, float]
    ribosome_density: pd.DataFrame
    prime_weights: Dict[str, float]


def load_tRNA_data(species: str) -> Dict[str, float]:
    """Return example tRNA abundance for a species.

    Real implementations would load empirical measurements.  Here a
    minimal dictionary provides deterministic behaviour for unit tests.
    """
    return {"AAA": 1.0, "TTT": 1.2, "GGG": 0.8}


def load_ribo_profiling(species: str) -> pd.DataFrame:
    """Return mock ribosome density data.

    In practice ribosome profiling yields transcript-level occupancy.
    These values mirror traffic on a highway where codons are road
    segments and density indicates slowdowns.
    """
    return pd.DataFrame(
        {
            "transcript": ["AAA", "TTT", "GGG"],
            "ribosome_density": [1.0, 0.8, 1.5],
        }
    )


def calculate_prime_weights(species: str) -> Dict[str, float]:
    """Return weighting factors for prime-aware optimisation."""
    return {"hydrophobicity": 0.35, "translation_speed": 0.45, "mutation_bias": 0.20}


def build_species_model(species: str) -> SpeciesModel:
    """Assemble tRNA and ribosome data into a model."""
    return SpeciesModel(
        tRNA_abundance=load_tRNA_data(species),
        ribosome_density=load_ribo_profiling(species),
        prime_weights=calculate_prime_weights(species),
    )


def split_into_codons(sequence: str) -> List[str]:
    """Divide a nucleotide sequence into codons (triplets)."""
    return [sequence[i : i + 3] for i in range(0, len(sequence), 3)]


def integrate_ribo_data(prime_products: pd.DataFrame, ribo_data: pd.DataFrame) -> pd.DataFrame:
    """Align prime products with ribosome density and compute stall scores."""
    merged = pd.merge(prime_products, ribo_data, left_on="codon", right_on="transcript")
    merged["stall_score"] = np.where(
        merged["ribosome_density"] > merged["prime_product"],
        merged["ribosome_density"] / merged["prime_product"],
        0,
    )
    return merged.sort_values("stall_score", ascending=False)


def prime_stall_index(ribosome_density: float, prime_product: float) -> float:
    """Return the Prime-Stall Index (PSI).

    PSI resembles a pressure gauge.  A value between 0.8 and 1.2
    indicates balanced traffic between ribosomes and prime codon
    composition.
    """
    if prime_product == 0:
        return float("inf")
    return log(ribosome_density) / prime_product


class PrimeGeneOptimizer:
    """Optimise gene sequences using prime mappings and species context."""

    def __init__(self, species: str = "e_coli") -> None:
        self.prime_map = {"A": 2, "C": 3, "G": 5, "T": 7}
        self.species_model = build_species_model(species)

    def predict_expression(self, sequence: str) -> float:
        """Predict expression level from prime composition and tRNA usage."""
        prime_score = sum(self.prime_map[b] for b in sequence) / len(sequence)
        codons = split_into_codons(sequence)
        tRNA_score = np.mean([
            self.species_model.tRNA_abundance.get(c, 0.0) for c in codons
        ])
        return 0.6 * tRNA_score + 0.4 * (100 / prime_score)

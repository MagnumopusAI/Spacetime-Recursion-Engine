from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Dict, List, Tuple, Union

from scipy.integrate import solve_ivp

import numpy as np
import pandas as pd

PRIME_MAP = {"A": 2, "C": 3, "G": 5, "T": 7}


def encode_prime_product(codon_or_counts: Union[str, Tuple[int, int, int, int]]) -> int:
    """Return the prime-product representation for a codon or base counts.

    Multiplying prime charges for each base mirrors how conserved charges
    add in a particle ensemble: order is irrelevant, only composition
    matters.  Accepts either a codon string or a base-count vector.
    """

    primes = (2, 3, 5, 7)
    if isinstance(codon_or_counts, str):
        product = 1
        for base in codon_or_counts:
            product *= PRIME_MAP[base]
        return product
    product = 1
    for p, n in zip(primes, codon_or_counts):
        product *= p ** n
    return product


def encode_base_counts(codon: str) -> Tuple[int, int, int, int]:
    """Return ``(n_A, n_C, n_G, n_T)`` for ``codon``.

    This vector resembles counting atoms in a molecule: each entry tracks
    how many of a particular nucleotide participate, satisfying a
    Preservation Constraint Equation for base composition.
    """

    counts = {"A": 0, "C": 0, "G": 0, "T": 0}
    for base in codon:
        counts[base] += 1
    return counts["A"], counts["C"], counts["G"], counts["T"]


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


def integrate_ribo_data(
    codon_features: pd.DataFrame, ribo_data: pd.DataFrame
) -> pd.DataFrame:
    """Align codon features with ribosome density and compute stall scores.

    ``codon_features`` may supply either a ``prime_product`` column or a
    ``base_counts`` column. When counts are present they are converted to
    prime products so the resulting stall scores are comparable across
    encodings.
    """

    if "prime_product" not in codon_features.columns:
        codon_features = codon_features.copy()
        codon_features["prime_product"] = codon_features["codon"].apply(
            encode_prime_product
        )
    merged = pd.merge(codon_features, ribo_data, left_on="codon", right_on="transcript")
    merged["stall_score"] = np.where(
        merged["ribosome_density"] > merged["prime_product"],
        merged["ribosome_density"] / merged["prime_product"],
        0,
    )
    return merged.sort_values("stall_score", ascending=False)


def prime_stall_index(
    ribosome_density: float, encoding: Union[float, Tuple[int, int, int, int]]
) -> float:
    """Return the Prime-Stall Index (PSI).

    ``encoding`` may be a prime-product scalar or a base-count vector.
    """

    prime_product = (
        encode_prime_product(encoding) if isinstance(encoding, (tuple, list)) else encoding
    )
    if prime_product == 0:
        return float("inf")
    return log(ribosome_density) / prime_product


class PrimeGeneOptimizer:
    """Optimise gene sequences using prime mappings and species context."""

    def __init__(self, species: str = "e_coli") -> None:
        self.prime_map = PRIME_MAP
        self.species_model = build_species_model(species)

    def predict_expression(self, sequence: str) -> float:
        """Predict expression level from prime composition and tRNA usage."""
        prime_score = sum(self.prime_map[b] for b in sequence) / len(sequence)
        codons = split_into_codons(sequence)
        tRNA_score = np.mean([
            self.species_model.tRNA_abundance.get(c, 0.0) for c in codons
        ])
        return 0.6 * tRNA_score + 0.4 * (100 / prime_score)


class PrimeAwareGeneOptimizer:
    """Optimise a gene sequence using analogue SAT dynamics."""

    def __init__(
        self,
        codons: list[list[tuple[str, float]]],
        n_vars: int,
        tAI_weights: list[list[float]],
        ribo_data: list[list[float]],
        C_m: float = 1.0,
        G_max: float = 0.1,
    ) -> None:
        self.codons = codons
        self.n_vars = n_vars
        self.tAI_weights = tAI_weights
        self.ribo_data = ribo_data
        self.C_m = C_m
        self.G_max = G_max
        rng = np.random.default_rng(0)
        self.V_init = rng.uniform(-1, 1, n_vars)
        self.W = rng.uniform(0, G_max, (n_vars, n_vars))
        np.fill_diagonal(self.W, 0)

    def ion_current(self, V: np.ndarray) -> np.ndarray:
        """Return stabilising ion currents.

        The voltages mimic charged codon states seeking a neutral
        configuration much like capacitors discharging toward zero.
        """

        return -0.1 * V

    def gap_junction_current(self, V: np.ndarray) -> np.ndarray:
        """Return neighbour-induced currents from ``self.W``."""

        return np.sum(self.W * (V[:, None] - V[None, :]), axis=1)

    def gene_loss(self, V: np.ndarray) -> float:
        """Evaluate codon quality against tAI and ribosome stalling."""

        loss = 0.0
        for i in range(self.n_vars):
            idx = int(np.clip(V[i], 0, len(self.codons[i]) - 1))
            tAI = self.tAI_weights[i][idx]
            pause = self.ribo_data[i][idx]
            loss += (1 - tAI) ** 2 + pause**2
        return float(loss)

    def external_current(self, V: np.ndarray) -> np.ndarray:
        """Return guidance currents pushing toward lower loss."""

        grad = np.zeros(self.n_vars)
        loss_val = self.gene_loss(V)
        for i in range(self.n_vars):
            grad[i] = -loss_val if V[i] > 0 else loss_val
        return grad

    def dynamics(self, t: float, V: np.ndarray) -> np.ndarray:
        """Differential update rule for codon voltages."""

        return (-1 / self.C_m) * (
            self.ion_current(V)
            + self.gap_junction_current(V)
            - self.external_current(V)
        )

    def solve(self, t_max: float = 10.0, tol: float = 1e-3):
        """Run the optimisation dynamics until the loss falls below ``tol``."""

        def event(_t, V):
            return self.gene_loss(V) - tol

        event.terminal = True
        sol = solve_ivp(self.dynamics, [0, t_max], self.V_init, events=event)
        V_final = sol.y[:, -1]
        codon_choices = [
            self.codons[i][int(np.clip(V_final[i], 0, len(self.codons[i]) - 1))][0]
            for i in range(self.n_vars)
        ]
        return codon_choices, self.gene_loss(V_final) < tol

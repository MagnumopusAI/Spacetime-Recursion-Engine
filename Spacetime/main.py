"""Entry point for the Spacetime Recursion Engine.

This script assembles a small lattice of domains and evaluates the
Preservation Constraint Equation (PCE) for each. Results are exported
using Sparse Prime Representation (SPR) to ``trigram_lattice.csv``.
"""

from pathlib import Path
from math import sqrt
import csv

from src.preservation import check_preservation
from src.torsion import TorsionField


PRIME_IDS = {
    "physics": 2,
    "biology": 3,
    "finance": 5,
    "cosmology": 7,
    "quantum": 11,
    "poincare": 13,
}


def generate_nodes():
    """Return a list of domain-specific ``(domain, sigma)`` tuples."""

    return [
        ("physics", 1.0),
        ("biology", 0.8),
        ("finance", 0.5),
        ("cosmology", 2.0),
        ("quantum", 1.2),
        ("poincare", 0.1),
    ]


def run_spacetime_engine(out_path: Path = Path("trigram_lattice.csv")) -> None:
    """Run the lattice evaluation and write an SPR encoded CSV."""

    tf = TorsionField(kappa=1.0)
    records = []

    for domain, sigma in generate_nodes():
        tau, ok = check_preservation(sigma)
        psi = [sigma, tau, 0.0, 0.0]
        torsion = tf.compute_torsion(psi)
        torsion_norm = float(abs(torsion).sum())
        records.append({
            "id": PRIME_IDS[domain],
            "sigma": round(sigma, 6),
            "tau": round(tau, 6),
            "torsion": round(torsion_norm, 6),
            "ok": int(ok),
        })

    with out_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "sigma", "tau", "torsion", "ok"])
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    run_spacetime_engine()

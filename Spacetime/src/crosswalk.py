"""Quadratic-Invariant Crosswalk utilities.

This module constructs a table summarizing invariants relevant to the
Beal-resonance narrative.  The input is a JSON file describing each
invariant, including whether the relation is quadratic in form.  The
output table lists the invariant name with discipline, the quadratic or
eigenmode expression, and the corresponding slot within the narrative.

Each helper is kept intentionally modular.  The overall design mirrors
the Preservation Constraint Equation (PCE) workflow where a simple rule
determines how components are sorted and assembled into a coherent
structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class Invariant:
    """Representation of a physical or mathematical invariant.

    Parameters mimic experimental cataloging: the *name* and *discipline*
    describe the observable, ``form`` captures the quadratic or eigenmode
    expression, and ``beal_slot`` links the invariant to its role in the
    Beal-resonance narrative.
    """

    name: str
    discipline: str
    form: str
    beal_slot: str
    quadratic: bool = False


@dataclass
class CrosswalkRow:
    """Single row of the crosswalk table."""

    name_disc: str
    form: str
    slot: str


def load_invariants(json_path: Path) -> list[dict]:
    """Load invariants from ``json_path`` and sort quadratic items first.

    Parameters
    ----------
    json_path:
        Location of the invariant definitions.

    Returns
    -------
    list[Invariant]
        Ordered invariants with quadratic relations leading.
    """

    with json_path.open() as f:
        raw = json.load(f)

    invariants = [Invariant(**inv) for inv in raw]
    invariants.sort(key=lambda inv: int(not inv.quadratic))
    return invariants


def build_crosswalk(invariants: Iterable[Invariant]) -> list[CrosswalkRow]:
    """Return a table summarizing the invariants.

    The table rows contain ``[name & discipline, quadratic form, slot]``.
    This structure mirrors how physical observables map into the
    Beal-resonance storyline.
    """

    rows: list[CrosswalkRow] = []
    for inv in invariants:
        name_disc = f"{inv.name} ({inv.discipline})"
        rows.append(CrosswalkRow(name_disc, inv.form, inv.beal_slot))
    return rows


def write_crosswalk_markdown(rows: Iterable[CrosswalkRow], out_path: Path) -> None:
    """Write the crosswalk table to ``out_path`` in Markdown format."""

    with out_path.open("w") as f:
        f.write(
            "| Invariant & Discipline | Quadratic/Eigenmode Form | Beal-Resonance Slot |\n"
        )
        f.write("| --- | --- | --- |\n")
        for row in rows:
            f.write(f"| {row.name_disc} | {row.form} | {row.slot} |\n")


def generate_crosswalk(json_path: Path, out_path: Path) -> None:
    """Convenience wrapper to create the Markdown table.

    This mirrors how experimental results are often processed: read the raw
    catalog of invariants, apply a simple ordering heuristic (quadratic forms
    first), then emit a table ready for inclusion in Section 2.3 of the
    manuscript.
    """

    invariants = load_invariants(json_path)
    rows = build_crosswalk(invariants)
    write_crosswalk_markdown(rows, out_path)


if __name__ == "__main__":  # pragma: no cover - manual utility
    repo_root = Path(__file__).resolve().parents[2]
    json_file = repo_root / "data" / "invariants.json"
    md_file = repo_root / "docs" / "quadratic_crosswalk.md"
    generate_crosswalk(json_file, md_file)


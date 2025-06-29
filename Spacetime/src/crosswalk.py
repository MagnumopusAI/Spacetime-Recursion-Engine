"""Quadratic-Invariant Crosswalk utilities.

This module constructs a table summarizing invariants relevant to the
Beal-resonance narrative. The input is a JSON file describing each
invariant, including whether the relation is quadratic in form.  The
output table lists the invariant name with discipline, the quadratic or
eigenmode expression, and the corresponding slot within the narrative.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def load_invariants(json_path: Path) -> list[dict]:
    """Load invariants from ``json_path`` and sort quadratic items first.

    Parameters
    ----------
    json_path:
        Location of the invariant definitions.

    Returns
    -------
    list[dict]
        Ordered invariants with quadratic relations leading.
    """

    with json_path.open() as f:
        invariants = json.load(f)

    invariants.sort(key=lambda inv: int(not inv.get("quadratic", False)))
    return invariants


def build_crosswalk(invariants: list[dict]) -> list[list[str]]:
    """Return a table summarizing the invariants.

    The table rows contain ``[name & discipline, quadratic form, slot]``.
    This structure mirrors how physical observables map into the
    Beal-resonance storyline.
    """

    rows: list[list[str]] = []
    for inv in invariants:
        name_disc = f"{inv['name']} ({inv['discipline']})"
        rows.append([name_disc, inv["form"], inv["beal_slot"]])
    return rows


def write_crosswalk_markdown(rows: list[list[str]], out_path: Path) -> None:
    """Write the crosswalk table to ``out_path`` in Markdown format."""

    with out_path.open("w") as f:
        f.write(
            "| Invariant & Discipline | Quadratic/Eigenmode Form | Beal-Resonance Slot |\n"
        )
        f.write("| --- | --- | --- |\n")
        for name_disc, form, slot in rows:
            f.write(f"| {name_disc} | {form} | {slot} |\n")


"""Cohomological rank evaluation utilities for symbolic memory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RankDiagnostic:
    """Summarize whether a collection of cohomology ranks satisfies the threshold.

    The diagnostic mirrors a structural engineer's inspection report: we compute
    the aggregate rank of the memory manifold and announce whether it is sturdy
    enough to store the symbolic pattern without collapsing topological features.
    """

    total_rank: int
    meets_threshold: bool


def evaluate_cohomological_rank(ranks: Iterable[int], threshold: int = 3) -> RankDiagnostic:
    """Return the total rank and compliance with the minimum threshold.

    Parameters
    ----------
    ranks:
        Iterable of individual Betti numbers for the memory complex.
    threshold:
        Minimum sum required before the memory state is admitted. The default of
        three reflects the need for a volumetric memory scaffold.
    """
    total = sum(int(value) for value in ranks)
    meets = total >= threshold
    return RankDiagnostic(total, meets)

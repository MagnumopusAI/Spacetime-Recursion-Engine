#!/usr/bin/env python3
"""ΔK Vector Projection heuristic for SMUG.

This module provides a compact mathematical analog of tracking solver
progress.  Imagine a rover traversing a shifting landscape: every
telemetry reading (variable activity, clause quality, database size)
forms a component of its *knowledge state vector* ``K``.  The rover's
stride between two steps is ``ΔK``; projecting this displacement onto a
scalar mirrors how far it truly advanced, akin to measuring the
Pythagorean distance between waypoints.  Although independent from the
Preservation Constraint Equation (PCE), this perspective keeps search
heuristics mindful of conserved quantities.
"""

from __future__ import annotations

import numpy as np


def define_k_vector(top_n_activities: list[float],
                    lbd_distribution: list[int],
                    db_size: int) -> np.ndarray:
    """Return a normalized knowledge state vector ``K``.

    Parameters
    ----------
    top_n_activities:
        VSIDS scores for the top variables, like energy readings from the
        most excited oscillators.
    lbd_distribution:
        Histogram of LBD scores for recent clauses, analogous to quality
        ratings of collected samples where lower scores indicate clearer
        signals.
    db_size:
        Count of learned clauses, similar to the storage fill level of a
        data recorder.

    Returns
    -------
    numpy.ndarray
        Concatenated and normalized vector capturing the solver state.
    """

    norm_activities = np.array(top_n_activities, dtype=float)
    if np.max(norm_activities) > 0:
        norm_activities = norm_activities / np.max(norm_activities)

    norm_lbd = np.array(lbd_distribution, dtype=float)
    if np.sum(norm_lbd) > 0:
        norm_lbd = norm_lbd / np.sum(norm_lbd)

    norm_db_size = np.array([min(db_size / 100000.0, 1.0)], dtype=float)

    k_vector = np.concatenate([norm_activities, norm_lbd, norm_db_size])
    return k_vector


def project_delta_k(delta_k_vector: np.ndarray) -> float:
    """Project ``ΔK`` onto a scalar progress measure.

    The L2 norm plays the role of a displacement operator: it reports the
    magnitude of change in the solver's knowledge state, much like a
    seismometer quantifying the strength of ground motion.

    Parameters
    ----------
    delta_k_vector:
        Difference ``K_t - K_{t-1}`` representing the latest stride of the
        search process.

    Returns
    -------
    float
        The magnitude of ``ΔK``.
    """

    return float(np.linalg.norm(delta_k_vector))


__all__ = ["define_k_vector", "project_delta_k"]


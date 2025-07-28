"""SU(4) commutation graph utilities.

This module constructs the 15 generators of the :math:`\mathfrak{su}(4)` Lie
algebra, computes their pairwise commutators and generates the adjacency matrix
encoding non-commuting interactions.

Real-world analogy
------------------
Each generator is like a fundamental control knob on a two-qubit quantum
computer. The adjacency matrix is a map showing which knobs interfere with each
other when turned simultaneously, mirroring the Heisenberg uncertainty
relationships.
"""
from __future__ import annotations

import numpy as np


def su4_generators() -> list[np.ndarray]:
    """Return the standard 15 generators of ``su(4)``.

    The matrices follow the generalized Gell--Mann construction with
    normalization chosen for unit trace orthogonality. This basis is common in
    particle physics and quantum information when describing four-level
    systems.
    """
    sqrt = np.sqrt
    zeros = np.zeros

    lam1 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    lam2 = np.array([[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    lam3 = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)

    lam4 = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    lam5 = np.array([[0, 0, -1j, 0], [0, 0, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    lam6 = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=complex)
    lam7 = np.array([[0, 0, 0, -1j], [0, 0, 0, 0], [0, 0, 0, 0], [1j, 0, 0, 0]], dtype=complex)

    lam8 = np.diag([1, 1, -2, 0]) / sqrt(3)

    lam9 = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=complex)
    lam10 = np.array([[0, 0, 0, 0], [0, 0, -1j, 0], [0, 1j, 0, 0], [0, 0, 0, 0]], dtype=complex)
    lam11 = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=complex)
    lam12 = np.array([[0, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 0, 0], [0, 1j, 0, 0]], dtype=complex)

    lam13 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    lam14 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=complex)
    lam15 = np.diag([1, 1, 1, -3]) / sqrt(6)

    return [
        lam1,
        lam2,
        lam3,
        lam4,
        lam5,
        lam6,
        lam7,
        lam8,
        lam9,
        lam10,
        lam11,
        lam12,
        lam13,
        lam14,
        lam15,
    ]


def commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the matrix commutator ``[a, b] = ab - ba``."""
    return a @ b - b @ a


def su4_adjacency_matrix(tol: float = 1e-9) -> np.ndarray:
    """Compute the adjacency matrix of the ``su(4)`` commutation graph.

    Parameters
    ----------
    tol:
        Numerical threshold for detecting non-zero commutators.

    Returns
    -------
    numpy.ndarray
        ``15 x 15`` symmetric matrix where ``1`` indicates a non-vanishing
        commutator between two generators.
    """
    gens = su4_generators()
    n = len(gens)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            comm = commutator(gens[i], gens[j])
            if np.any(np.abs(comm) > tol):
                adj[i, j] = adj[j, i] = 1
    return adj


def adjacency_spectrum(adj: np.ndarray) -> np.ndarray:
    """Return eigenvalues of a commutation adjacency matrix."""
    vals = np.linalg.eigvals(adj)
    return np.sort(vals)

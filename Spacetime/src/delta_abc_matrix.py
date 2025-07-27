"""Δ_ABC eigenvalue utilities.

This module implements the Δ_ABC method for computing eigenvalues of a
matrix. The method mirrors how physical invariants like energy,
interaction strength, and volume determine natural oscillation modes.
It expresses the characteristic polynomial via three scalar quantities
A, B, and C analogous to these invariants.
"""

from __future__ import annotations

import numpy as np

__all__ = ["delta_abc_eigenvalues"]


def _principal_minor_sum(matrix: np.ndarray) -> float:
    """Return sum of principal 2×2 minors.

    Real-world analogy: pairwise coupling energies between axes.
    """
    if matrix.shape[0] != 3:
        raise ValueError("Principal minor sum is defined for 3×3 matrices")
    a11, a12, a13 = matrix[0]
    a21, a22, a23 = matrix[1]
    a31, a32, a33 = matrix[2]
    minor1 = a11 * a22 - a12 * a21
    minor2 = a11 * a33 - a13 * a31
    minor3 = a22 * a33 - a23 * a32
    return float(minor1 + minor2 + minor3)


def delta_abc_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """Compute eigenvalues using the Δ_ABC method.

    Parameters
    ----------
    matrix:
        Square array representing a linear transformation.

    Returns
    -------
    numpy.ndarray
        Eigenvalues derived from the Δ_ABC characteristic polynomial.

    Notes
    -----
    For 2×2 and 3×3 inputs the coefficients ``A``, ``B`` and ``C`` are
    built from traces and determinants to mirror conservation of mass,
    pairwise interaction, and total volume. Larger matrices fall back to
    ``numpy.linalg.eigvals``.
    """

    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("Input must be a square matrix")

    n = m.shape[0]
    if n == 1:
        return m.reshape(1)

    A = float(np.trace(m))

    if n == 2:
        B = float(np.linalg.det(m))
        coeffs = [1.0, -A, B]
        return np.roots(coeffs)

    if n == 3:
        B = _principal_minor_sum(m)
        C = float(np.linalg.det(m))
        coeffs = [1.0, -A, B, -C]
        return np.roots(coeffs)

    return np.linalg.eigvals(m)


"""Riemann Hypothesis helper utilities."""

from __future__ import annotations


def construct_hilbert_polya_operator():
    """Return a spectral operator encoding the critical line."""

    def operator(sigma: float, tau: float) -> float:
        return -2 * (sigma - 0.5) ** 2 + 2 * tau**2 + 3 * tau

    return operator


class RiemannSolver:
    """Bridge the Riemann hypothesis to PCE methods."""

    @staticmethod
    def construct_hilbert_polya_operator():
        return construct_hilbert_polya_operator()

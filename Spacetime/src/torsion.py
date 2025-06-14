"""Torsion Field utilities.

This module contains a simplified representation of torsion dynamics used by the
Spacetime Recursion Engine.  The main purpose is to provide a numerically stable
approximation of the torsion tensor predicted by the SMUG framework.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def model_torsion_flow(spin_density, beta, beta_tilde, F, r=10.0):
    """Estimate the magnitude of a torsion-induced collapse force.

    Parameters
    ----------
    spin_density:
        The spinor field density.
    beta:
        Primary coupling constant.
    beta_tilde:
        Secondary coupling constant.
    F:
        Radial profile, such as ``lambda r: r**2``.
    r:
        Radius value used to evaluate ``F``.

    Returns
    -------
    float
        Approximated collapse force.
    """

    return beta * beta_tilde * spin_density * F(r) * r


@dataclass
class TorsionField:
    """Compute torsion tensors from spinor bilinears.

    The implementation follows the schematic form of

    ``T^λ_{μν} = (κ^2/2) ε^{λρστ} \bar{ψ} γ_ρ γ^5 ψ g_{σμ} g_{τν}``

    where ``κ`` is an effective coupling constant.  The result is a ``4×4×4``
    tensor.  This simplified version uses canonical Dirac matrices and a flat
    Minkowski metric.
    """

    kappa: float = 1.0

    def _gamma_matrices(self) -> list[np.ndarray]:
        """Return Dirac gamma matrices in the Dirac representation."""

        g0 = np.diag([1, 1, -1, -1])
        g1 = np.array([[0, 0, 0, 1],
                       [0, 0, 1, 0],
                       [0, -1, 0, 0],
                       [-1, 0, 0, 0]], dtype=complex)
        g2 = np.array([[0, 0, 0, -1j],
                       [0, 0, 1j, 0],
                       [0, 1j, 0, 0],
                       [-1j, 0, 0, 0]], dtype=complex)
        g3 = np.array([[0, 0, 1, 0],
                       [0, 0, 0, -1],
                       [-1, 0, 0, 0],
                       [0, 1, 0, 0]], dtype=complex)
        return [g0, g1, g2, g3]

    def _gamma5(self) -> np.ndarray:
        g0, g1, g2, g3 = self._gamma_matrices()
        return 1j * g0 @ g1 @ g2 @ g3

    def _levi_civita(self) -> np.ndarray:
        """Return the 4D Levi-Civita symbol."""

        eps = np.zeros((4, 4, 4, 4))
        import itertools

        for perm in itertools.permutations(range(4)):
            sign = 1
            perm_list = list(perm)
            for a in range(3):
                for b in range(a + 1, 4):
                    if perm_list[a] > perm_list[b]:
                        sign *= -1
            eps[perm] = sign
        return eps

    def compute_torsion(self, psi: np.ndarray) -> np.ndarray:
        """Compute the torsion tensor for a given spinor ``psi``.

        Parameters
        ----------
        psi:
            Four-component spinor array.

        Returns
        -------
        numpy.ndarray
            A ``4×4×4`` torsion tensor.
        """

        psi = np.asarray(psi, dtype=complex).reshape(4, 1)
        g = np.diag([1, -1, -1, -1])

        gamma = self._gamma_matrices()
        gamma5 = self._gamma5()
        psi_bar = psi.conj().T @ gamma[0]

        bilinear = np.array([
            float((psi_bar @ gamma[r] @ gamma5 @ psi).real)
            for r in range(4)
        ])

        eps = self._levi_civita()
        torsion = np.zeros((4, 4, 4))

        for lam in range(4):
            for mu in range(4):
                for nu in range(4):
                    s = 0.0
                    for rho in range(4):
                        for sigma in range(4):
                            for tau in range(4):
                                s += (
                                    eps[lam, rho, sigma, tau]
                                    * bilinear[rho]
                                    * g[sigma, mu]
                                    * g[tau, nu]
                                )
                    torsion[lam, mu, nu] = (self.kappa**2 / 2.0) * s

        return torsion



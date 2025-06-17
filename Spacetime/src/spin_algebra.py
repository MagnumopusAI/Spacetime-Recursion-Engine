"""spin(5,1) Clifford algebra utilities."""

from __future__ import annotations

from sympy import Matrix, eye, kronecker_product, I


class Spin51Algebra:
    r"""Construct generators for ``\mathfrak{spin}(5,1)``."""

    def __init__(self) -> None:
        self.generators = self.construct_clifford_algebra()

    def construct_clifford_algebra(self) -> dict[str, Matrix]:
        """Return 15 generators for spin(5,1) acting on a 16D spinor space."""

        generators: dict[str, Matrix] = {}
        for i in range(6):
            for j in range(i + 1, 6):
                gen = self._gamma_wedge(i, j)
                generators[f"sigma_{i}{j}"] = gen
        return generators

    def _gamma_matrices(self) -> list[Matrix]:
        """Return 16×16 gamma matrices for ``Cl(5,1)``."""

        # 4D Dirac matrices
        g0 = Matrix.diag(1, 1, -1, -1)
        g1 = Matrix([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]])
        g2 = Matrix(
            [[0, 0, 0, -I], [0, 0, I, 0], [0, I, 0, 0], [-I, 0, 0, 0]]
        )
        g3 = Matrix([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]])

        base = [g0, g1, g2, g3]
        # Expand to 16×16 using Kronecker product with identity
        expanded = [kronecker_product(eye(4), g) for g in base]
        # Additional two matrices for 6D Clifford algebra
        g4 = kronecker_product(Matrix([[0, 1], [-1, 0]]), eye(8))
        g5 = kronecker_product(Matrix([[1, 0], [0, -1]]), eye(8))
        expanded.extend([g4, g5])
        return expanded

    def _gamma_wedge(self, i: int, j: int) -> Matrix:
        """Construct ``γ_i ∧ γ_j`` for ``Cl(5,1)``."""

        gamma = self._gamma_matrices()
        return gamma[i] * gamma[j] - gamma[j] * gamma[i]

    def select_physical_eigenmode(self, states: list[dict]) -> dict | None:
        r"""Return the state with ``\lambda = 4`` if present."""

        for st in states:
            if st.get("lambda") == 4:
                return st
        return None

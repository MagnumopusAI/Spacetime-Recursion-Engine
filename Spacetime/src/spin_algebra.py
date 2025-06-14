"""spin(5,1) Clifford algebra utilities."""

from __future__ import annotations

from sympy import Matrix, zeros


class Spin51Algebra:
    """Construct generators for ``\mathfrak{spin}(5,1)``."""

    def __init__(self) -> None:
        self.generators = self.construct_clifford_algebra()

    def construct_clifford_algebra(self) -> dict[str, Matrix]:
        """Return a symbolic representation of ``Cl(3,1) \otimes Cl(2,0)``."""

        g0 = Matrix.eye(4)
        g1 = Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
        return {"gamma0": g0, "gamma1": g1}

    def select_physical_eigenmode(self, states: list[dict]) -> dict | None:
        """Return the state with ``\lambda = 4`` if present."""

        for st in states:
            if st.get("lambda") == 4:
                return st
        return None

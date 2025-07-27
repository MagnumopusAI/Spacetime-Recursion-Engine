"""SMUG-themed 3-SAT architecture utilities."""

from __future__ import annotations

import random
from typing import List, Tuple, Dict
import numpy as np

from .millennium.p_vs_np import resolve_p_vs_np

Clause = List[Tuple[int, bool]]
Instance = List[Clause]


def generate_3sat(n: int, m: int) -> Instance:
    """Return a random 3-SAT instance with ``n`` variables and ``m`` clauses.

    Each clause mimics a local interaction in a lattice, composed of three
    ``(variable, value)`` pairs.
    """

    instance: Instance = []
    for _ in range(m):
        vars_sample = random.sample(range(1, n + 1), 3)
        clause = [(v, random.choice([True, False])) for v in vars_sample]
        instance.append(clause)
    return instance


def _to_solver_format(instance: Instance) -> List[Tuple[int, int, int]]:
    """Convert ``Instance`` to signed-literal representation."""

    return [tuple(v if val else -v for v, val in clause) for clause in instance]


class SMUGArchitect:
    """Coordinate simple 3-SAT demonstrations using SMUG analogies."""

    def __init__(self, source: str = "Local Stellar Group") -> None:
        self.source = source
        self.blueprints: Dict[str, Dict[str, object]] = {}
        self.lattice: Dict[str, object] = {"type": "SU(4)", "dim": 4}

    def print_imperatives_map(self) -> None:
        """Display an imaginative map linking tools to big-picture goals."""

        print("SeussBot's Imperative Map:\n")
        print("- Longevity Escape Velocity: Dirac-field inspired resilience.")
        print("- Redistribute Wealth: prime systems ensuring fairness.")
        print("- Increase Understanding: bridging math and physics.")

    def print_interconnections(self) -> None:
        """Print lattice information and number of designed artifacts."""

        print("SeussBot's Interconnections:")
        print(f"Lattice type: {self.lattice['type']} with dimension {self.lattice['dim']}")
        print(f"Blueprints stored: {len(self.blueprints)}")

    def compute_artifact(
        self,
        artifact_id: str,
        instance: Instance,
        mode: str = "formalist",
        *,
        upsilon: float = np.pi,
        n_qubits: int = 3,
    ) -> Dict[int, bool] | None:
        """Solve ``instance`` via :func:`resolve_p_vs_np` and store the result.

        The parameters ``upsilon`` and ``n_qubits`` evoke physical knobs while
        the underlying search respects the Preservation Constraint Equation by
        delegating to the core solver.
        """

        literals = _to_solver_format(instance)
        result = resolve_p_vs_np(literals, observer_mode=mode)
        self.blueprints[artifact_id] = {
            "type": "3-SAT_solution",
            "mode": mode,
            "result": result,
            "upsilon": upsilon,
            "n_qubits": n_qubits,
        }
        return result

    def verify_assignment(self, clauses: Instance, assignment: Dict[int, bool] | None) -> bool:
        """Return ``True`` if ``assignment`` satisfies ``clauses``."""

        if not assignment:
            return False
        for clause in clauses:
            if not any(assignment.get(var) == val for var, val in clause):
                return False
        return True

    def instantiate_computational_manifold(self, n: int) -> Dict[str, object]:
        """Initialize an ``n``-dimensional SU(4) lattice."""

        volume = n ** 4
        self.lattice = {"type": "SU(4)", "dim": n, "volume": volume}
        return self.lattice


def auto_verify_3sat(
    instance: Instance,
    *,
    upsilon: float = np.pi,
    n_qubits: int = 3,
) -> Tuple[Dict[int, bool] | None, Dict[int, bool] | None]:
    """Solve ``instance`` in both observer modes and verify each result."""

    architect = SMUGArchitect()
    formalist_result = architect.compute_artifact(
        "P023_formalist", instance, "formalist", upsilon=upsilon, n_qubits=n_qubits
    )
    smug_result = architect.compute_artifact(
        "P023_smug", instance, "smug", upsilon=upsilon, n_qubits=n_qubits
    )
    assert architect.verify_assignment(instance, formalist_result)
    assert architect.verify_assignment(instance, smug_result)
    return formalist_result, smug_result


if __name__ == "__main__":
    demo_instance = generate_3sat(3, 3)
    auto_verify_3sat(demo_instance)

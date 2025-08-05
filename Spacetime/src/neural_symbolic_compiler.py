"""Neural-symbolic compiler aware of the cosmic tau rhythm.

This module translates logical clause projectors into gate sequences while
respecting the dynamic reliability threshold imposed by the phase energy
parameter :math:`\tau(T)`.  When :math:`\tau` is large, mistakes are costly and
the compiler demands purer qubits; when :math:`\tau` is small, it permits a
more adventurous use of noisy hardware.  The overall structure honors the
Preservation Constraint Equation by separating the neural heuristic (Right
Brain) from the symbolic rewrite (Left Brain).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

Clause = List[Tuple[int, bool]]


@dataclass
class ClauseProjector:
    """Container for a logical clause projector.

    The ``name`` labels the clause within a larger SAT instance while ``clause``
    stores a list of ``(variable, value)`` pairs.  Physically, each pair encodes
    a desired spin orientation that the compiler must imprint onto qubits.
    """

    name: str
    clause: Clause


class NeuralSymbolicCompiler:
    """Blend neural intuition with symbolic rigor to compile clause projectors.

    Parameters
    ----------
    architect:
        Object supplying hardware calibration data.  It must implement
        ``get_hardware_calibration_data`` returning a mapping of qubit names to
        reliability scores in ``[0, 1]``.
    """

    def __init__(self, architect) -> None:
        self.architect = architect
        self.holographic_blueprints: Dict[str, List[Tuple[str, ...]]] = {}

    # ------------------------------------------------------------------
    # Right-brain: heuristic qubit selection tuned by tau(T)
    # ------------------------------------------------------------------
    def _dynamic_threshold(self, tau: float) -> float:
        """Return the reliability threshold ``0.80 + 0.1 * tau``.

        A rising :math:`\tau` acts like increasing pressure on a spacecraft:
        the margin for error shrinks, so the allowed hardware noise decreases.
        """

        return 0.80 + 0.1 * tau

    def _neural_guess(
        self, clause: Clause, hardware_state: Dict[str, float], tau: float
    ) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        """Suggest a qubit map and template consistent with the tau threshold."""

        threshold = self._dynamic_threshold(tau)
        noisy_qubits = {q for q, r in hardware_state.items() if r < threshold}
        available = [q for q in hardware_state if q not in noisy_qubits]

        clause_len = len(clause)
        if clause_len <= 3:
            if len(available) < clause_len:
                return None, None
            qubit_map = {f"x{i+1}": q for i, q in enumerate(available[:clause_len])}
            template = "3-QUBIT-PARITY-CHECK"
        elif clause_len == 5:
            if len(available) < 6:  # five variables plus one ancilla
                return None, None
            qubit_map = {f"x{i+1}": q for i, q in enumerate(available[:5])}
            qubit_map["ancilla"] = available[5]
            template = "5-TO-3-FRAGMENTATION"
        else:  # pragma: no cover - enforced by explicit design
            raise NotImplementedError(
                "Only clauses of length ≤3 or exactly 5 are supported."
            )
        return qubit_map, template

    # ------------------------------------------------------------------
    # Left-brain: deterministic symbolic rewrite
    # ------------------------------------------------------------------
    def _symbolic_rewrite(
        self, clause: Clause, qubit_map: Dict[str, str], template: str
    ) -> List[Tuple[str, ...]]:
        """Translate ``template`` into a concrete gate sequence.

        The gates are expressed as tuples ``(gate, *qubits)`` for clarity,
        analogous to a string of musical notes in a score.
        """

        if template == "5-TO-3-FRAGMENTATION":
            seq = [
                ("Toffoli", qubit_map["x1"], qubit_map["x2"], qubit_map["ancilla"]),
                ("CNOT", qubit_map["x3"], qubit_map["ancilla"]),
                ("Toffoli", qubit_map["x4"], qubit_map["x5"], qubit_map["ancilla"]),
                ("CNOT", qubit_map["x3"], qubit_map["ancilla"]),
                ("Toffoli", qubit_map["x1"], qubit_map["x2"], qubit_map["ancilla"]),
            ]
        elif template == "3-QUBIT-PARITY-CHECK":
            seq = [
                ("CNOT", qubit_map["x1"], qubit_map["x3"]),
                ("CNOT", qubit_map["x2"], qubit_map["x3"]),
            ]
        else:  # pragma: no cover - template is validated upstream
            raise ValueError(f"Unknown template: {template}")
        return seq

    # ------------------------------------------------------------------
    # Full compilation pipeline
    # ------------------------------------------------------------------
    def compile(self, clause_projector: ClauseProjector, tau: float):
        """Compile ``clause_projector`` under a given ``tau`` value.

        Returns
        -------
        list[tuple[str, ...]] | None
            Gate sequence if sufficient reliable qubits exist, otherwise
            ``None`` signalling that the compiler declined the risky attempt.
        """

        hardware_state = self.architect.get_hardware_calibration_data()
        qubit_map, template = self._neural_guess(
            clause_projector.clause, hardware_state, tau
        )
        if not qubit_map:
            return None
        sequence = self._symbolic_rewrite(clause_projector.clause, qubit_map, template)
        key = f"{clause_projector.name}_tau_{tau:.2f}"
        self.holographic_blueprints[key] = sequence
        return sequence

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.smug_os import (
    build_smug_qiskit_ansatz,
    generate_complexity_wormhole,
    launch_clifford_pnp_oracle,
    simulate_smug_annihilation,
)


def test_simulate_smug_annihilation_prunes_contradictions():
    clauses = [[1, -1, 2], [1, 2], [-2, 3]]
    pruned = simulate_smug_annihilation(clauses)
    assert [1, -1, 2] not in pruned
    assert len(pruned) == 2


def test_build_smug_qiskit_ansatz_returns_valid_objects():
    clauses = [[1, -2], [2, 3]]
    qc, ham = build_smug_qiskit_ansatz(clauses)
    assert qc.num_qubits == 3
    # Hamiltonian should be defined over the same number of qubits
    assert ham.num_qubits == 3
    assert len(ham.oplist) >= 2


def test_generate_complexity_wormhole_mapping():
    mapping = generate_complexity_wormhole("SAT", "CLIQUE")
    assert mapping["source"] == "SAT"
    assert mapping["target"] == "CLIQUE"
    assert mapping["bridge_type"].startswith("Hodge_")


def test_launch_clifford_pnp_oracle_builds_circuit():
    clauses = [[1, 2], [-1, 2]]
    qc = launch_clifford_pnp_oracle(clauses)
    assert qc.num_qubits == 2
    assert qc.num_clbits == 2


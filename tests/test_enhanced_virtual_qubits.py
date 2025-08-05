import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sympy as sp

from Spacetime.src.enhanced_virtual_qubits import (
    SUGenerator,
    VirtualQubit,
    ClauseGraph,
    NeuralSymbolicSystem,
    deep_clifford_walk,
    run_smug_simulation,
)
from Spacetime.src.preservation import check_preservation


def test_su_generator_traceless():
    gens = SUGenerator.generate(3)
    assert len(gens) == 8
    for g in gens:
        assert sp.Matrix(g).trace() == 0


def test_virtual_qubit_gate_application():
    vq = VirtualQubit(2)
    X = sp.Matrix([[0, 1], [1, 0]])
    vq.apply_gate(X)
    assert vq.state == sp.Matrix([0, 1])


def test_clause_graph_edges():
    g = ClauseGraph([[1, -2], [2, 3]])
    assert g.edges[1] == [0]
    assert set(g.edges[2]) == {0, 1}


def test_hamiltonian_pce():
    nsys = NeuralSymbolicSystem(4)
    clauses = [[1, -2, 3]]
    H = nsys.construct_global_hamiltonian(clauses)
    diag_val = H[0, 0]
    sigma = len(clauses[0]) / nsys.dim
    tau, ok = check_preservation(sigma)
    assert ok
    assert sp.N(diag_val) == sp.N(tau)


def test_run_smug_simulation_keys():
    clauses = [[1, -2, 3], [-1, 2]]
    result = run_smug_simulation(clauses, dim=4, iters=1)
    assert set(result) == {"curvature", "qpe_spectrum", "sheaf_sections"}
    assert result["sheaf_sections"] > 0


def test_apply_xor_two_bit_flip():
    nsys = NeuralSymbolicSystem(4)
    U = nsys.apply_xor()
    assert U[0, 3] == 1
    assert U[3, 0] == 1
    assert U[1, 2] == 1
    assert U[2, 1] == 1
    assert U[0, 1] == 0


def test_symmetry_projection_z2():
    nsys = NeuralSymbolicSystem(4)
    P = nsys.apply_symmetry_projection()
    diag = [P[i, i] for i in range(4)]
    assert diag == [1, 0, 0, 1]


def test_deep_clifford_walk_measure():
    clauses = [[1, -2], [2, 3]]
    m = deep_clifford_walk(clauses, dim=4)
    assert m == 0

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Spacetime'))

from src.cohomology import SheafSAT, QuantumReasoningAgent


def test_cech_cohomology_simple():
    variables = ['x1', 'x2']
    clauses = [[('x1', 1)], [('x2', 0)]]
    sat = SheafSAT(variables, clauses)
    ranks = sat.compute_cech_cohomology()
    assert ranks[0] >= 1
    # No variable overlap => first cohomology should vanish
    assert ranks.get(1, 0) == 0


def test_quantum_reasoning_agent():
    variables = ['a', 'b']
    clauses = [[('a', 1), ('b', 0)], [('a', 0), ('b', 1)]]
    agent = QuantumReasoningAgent()
    report = agent.process(variables, clauses)
    assert 'Quantum Cohomology Report:' in report
    assert 'Mode tag' in report


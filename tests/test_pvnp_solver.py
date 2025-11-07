import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.pvnp_solver import (  # noqa: E402
    ComplexityValidator,
    SATToLatticeMapper,
    SolutionExtractor,
    SU4LatticeMinimizer,
    ThreeSATInstance,
    generate_test_3sat,
    solve_3sat_via_su4_lattice,
)


def _simple_sat_instance() -> ThreeSATInstance:
    variables = ("x1", "x2", "x3")
    clauses = ((1, -2, 3), (1, 2, -3))
    return ThreeSATInstance(variables=variables, clauses=clauses)


def test_mapper_assigns_unique_coordinates():
    sat_instance = _simple_sat_instance()
    mapper = SATToLatticeMapper()
    lattice = mapper.encode_3sat_instance(sat_instance)

    coords = list(lattice.var_positions.values())
    assert len(coords) == len(set(coords))
    assert lattice.lattice_size >= 1
    assert all(isinstance(op, np.ndarray) and op.shape == (4, 4) for op in lattice.clause_operators.values())


def test_hmc_minimizer_records_action_history():
    sat_instance = _simple_sat_instance()
    mapper = SATToLatticeMapper()
    lattice = mapper.encode_3sat_instance(sat_instance)

    minimizer = SU4LatticeMinimizer(lattice, seed=0)
    ground_state = minimizer.hybrid_monte_carlo_minimization(steps=5, epsilon=0.01)

    assert ground_state is not None
    assert len(minimizer.action_history) == 5
    assert all(np.isfinite(value) for value in minimizer.action_history)


def test_solution_extraction_matches_evaluation():
    sat_instance = _simple_sat_instance()
    mapper = SATToLatticeMapper()
    lattice = mapper.encode_3sat_instance(sat_instance)

    minimizer = SU4LatticeMinimizer(lattice, seed=1)
    ground_state = minimizer.hybrid_monte_carlo_minimization(steps=4, epsilon=0.02)

    extractor = SolutionExtractor(ground_state)
    assignments = extractor.extract_sat_assignment()

    assert extractor.verify_solution(sat_instance, assignments)
    assert np.isfinite(extractor.compute_topological_charge())


def test_full_pipeline_finds_solution():
    sat_instance = generate_test_3sat(3)
    solution, metadata = solve_3sat_via_su4_lattice(
        sat_instance, method="hmc", minimizer_kwargs={"steps": 4, "epsilon": 0.02}
    )

    assert sat_instance.evaluate(solution)
    assert metadata["satisfiable"] is True
    assert metadata["method"] == "hmc"


def test_complexity_validator_returns_finite_exponent():
    validator = ComplexityValidator()
    exponent = validator.benchmark_scaling(max_vars=3, step=2, seed=123)
    assert np.isfinite(exponent)


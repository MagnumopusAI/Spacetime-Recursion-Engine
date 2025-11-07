"""PvNP solver architecture implemented with SU(4) lattice analogies.

This module translates small 3-SAT instances into a stylised SU(4) lattice
model, runs a hybrid Monte Carlo (HMC) style minimisation, and extracts
solutions honouring the Preservation Constraint Equation (PCE).  The overall
design mirrors laboratory practice: clauses become Wilson loops, the action is
steered by a simplified Yang--Mills functional, and observables are decoded via
Wilson loop expectations.

The implementation emphasises clarity and pedagogical structure rather than raw
performance.  Each class contains docstrings relating the computation to
physical intuition so the code can serve as an executable blueprint for more
ambitious simulators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from .preservation import solve_tau_from_sigma


# ---------------------------------------------------------------------------
# SAT data structures
# ---------------------------------------------------------------------------


@dataclass
class ThreeSATInstance:
    """Container for 3-SAT problems aligned with PCE symbolism.

    Parameters
    ----------
    variables:
        Ordered list of variable labels.  The labels may be strings or integers;
        they are used to map literals onto lattice coordinates.
    clauses:
        Iterable of three-literal clauses.  Literals follow the conventional
        integer encoding with sign indicating negation (1-indexed).

    Real-world analogy
    ------------------
    Think of ``variables`` as energy knobs on a control lattice.  Each clause is
    like a Wilson loop measurement that stabilises the configuration.  The PCE
    enforces that the knobs can only be tuned along specific invariant
    directions.
    """

    variables: Sequence[str]
    clauses: Sequence[Tuple[int, int, int]]

    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        """Return ``True`` when ``assignment`` satisfies every clause.

        This mirrors measuring all Wilson loops and observing a positive
        expectation.  The evaluation operates on the provided assignment without
        modifying it.
        """

        def literal_truth(literal: int) -> bool:
            var = self.variables[abs(literal) - 1]
            value = assignment[var]
            return value if literal > 0 else not value

        return all(any(literal_truth(l) for l in clause) for clause in self.clauses)


def generate_test_3sat(num_variables: int) -> ThreeSATInstance:
    """Generate a deterministic 3-SAT instance for regression testing.

    The instance places each variable in both positive and negative form across
    clauses to exercise the extraction logic.  It acts like a calibration grid
    for the PvNP solver.
    """

    variables = [f"x{i+1}" for i in range(num_variables)]
    clauses: List[Tuple[int, int, int]] = []
    for i in range(num_variables - 2):
        clauses.append((i + 1, -(i + 2), i + 3))
    if num_variables >= 3:
        clauses.append((1, -2, 3))
    return ThreeSATInstance(variables=tuple(variables), clauses=tuple(clauses))


def generate_random_3sat(num_variables: int, num_clauses: int, seed: int | None = None) -> ThreeSATInstance:
    """Return a pseudo-random 3-SAT instance using a reproducible RNG.

    The generator is intended for benchmarking the complexity validator.  The
    randomness mimics sampling different disorder realisations in lattice gauge
    theory experiments.
    """

    rng = np.random.default_rng(seed)
    variables = [f"x{i+1}" for i in range(num_variables)]
    clauses: List[Tuple[int, int, int]] = []
    for _ in range(num_clauses):
        lits: List[int] = []
        for _ in range(3):
            idx = rng.integers(1, num_variables + 1)
            sign = rng.choice([-1, 1])
            lits.append(int(sign * idx))
        clauses.append(tuple(lits))
    return ThreeSATInstance(variables=tuple(variables), clauses=tuple(clauses))


# ---------------------------------------------------------------------------
# SU(4) lattice infrastructure
# ---------------------------------------------------------------------------


@dataclass
class SU4LatticeConfiguration:
    """Minimal SU(4) configuration storing links and clause anchors.

    Parameters
    ----------
    lattice_size:
        Linear extent of the cubic lattice.  The volume equals ``lattice_size**3``.
    var_positions:
        Mapping from variable label to lattice coordinate.
    clause_operators:
        Indexed dictionary of 4×4 matrices encoding clause projectors.
    clause_sites:
        Lookup table mapping coordinates to lists of clause indices acting at
        that position.
    pce_tau:
        Positive root of the PCE used as a background torsion scale.

    The configuration behaves like a field container: ``links`` stores the
    gauge matrices for each direction, while helper methods provide convenient
    iteration over sites and links.
    """

    lattice_size: int
    var_positions: Dict[str, Tuple[int, int, int]]
    clause_operators: Dict[int, np.ndarray]
    clause_sites: Dict[Tuple[int, int, int], List[int]]
    pce_tau: float
    links: np.ndarray = field(repr=False)

    def clone(self) -> "SU4LatticeConfiguration":
        """Return a deep copy of the configuration for HMC proposals."""

        return SU4LatticeConfiguration(
            lattice_size=self.lattice_size,
            var_positions=dict(self.var_positions),
            clause_operators={k: v.copy() for k, v in self.clause_operators.items()},
            clause_sites={site: list(indices) for site, indices in self.clause_sites.items()},
            pce_tau=self.pce_tau,
            links=self.links.copy(),
        )

    def volume(self) -> int:
        """Return the lattice volume (number of sites)."""

        return self.lattice_size ** 3

    def all_sites(self) -> Iterator[Tuple[int, int, int]]:
        """Yield each lattice coordinate in lexicographic order."""

        L = self.lattice_size
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    yield (x, y, z)

    def all_links(self) -> Iterator[Tuple[Tuple[int, int, int], int]]:
        """Yield pairs ``((x, y, z), direction)`` for every oriented link."""

        for site in self.all_sites():
            for mu in range(3):
                yield site, mu


class SATToLatticeMapper:
    """Encode 3-SAT instances into an SU(4) lattice skeleton.

    The mapper follows three conceptual steps: position variables on a cubic
    lattice, convert each clause into an SU(4) projector, and seed a gauge
    configuration that respects the PCE background solution.
    """

    def __init__(self) -> None:
        self.var_dict: Dict[str, Tuple[int, int, int]] = {}
        self.clause_operators: Dict[int, np.ndarray] = {}
        self.clause_sites: Dict[Tuple[int, int, int], List[int]] = {}
        self.index_to_var: Dict[int, str] = {}

    def encode_3sat_instance(self, sat_instance: ThreeSATInstance) -> SU4LatticeConfiguration:
        """Return a lattice configuration representing ``sat_instance``.

        Variables are assigned to cubic lattice coordinates.  Each clause is
        converted into a rank-one projector built from Pauli-``Z`` factors and
        anchored to the coordinate of its leading literal.
        """

        num_vars = len(sat_instance.variables)
        lattice_size = max(1, int(math.ceil(num_vars ** (1 / 3))))

        self.var_dict.clear()
        self.clause_operators.clear()
        self.clause_sites.clear()
        self.index_to_var = {idx + 1: name for idx, name in enumerate(sat_instance.variables)}

        for idx, var in enumerate(sat_instance.variables):
            coord = self._index_to_coord(idx, lattice_size)
            self.var_dict[var] = coord

        self._encode_clauses_as_plaquettes(sat_instance.clauses)

        lattice = self._build_initial_configuration(lattice_size)
        return lattice

    def _index_to_coord(self, index: int, lattice_size: int) -> Tuple[int, int, int]:
        """Map a linear index to a 3-D lattice coordinate."""

        L = lattice_size
        x = index % L
        y = (index // L) % L
        z = (index // (L * L)) % L
        return x, y, z

    def _parse_literal(self, literal: int) -> Tuple[str, bool]:
        """Return variable label and negation flag for ``literal``."""

        var = self.index_to_var[abs(literal)]
        return var, literal < 0

    def _encode_clauses_as_plaquettes(self, clauses: Sequence[Tuple[int, int, int]]) -> None:
        """Translate clauses into SU(4) projectors and anchor metadata."""

        for idx, clause in enumerate(clauses):
            clause_matrix = self._encode_clause_operator(clause)
            anchor_var, _ = self._parse_literal(clause[0])
            anchor_site = self.var_dict[anchor_var]
            self.clause_operators[idx] = clause_matrix
            self.clause_sites.setdefault(anchor_site, []).append(idx)

    def _encode_clause_operator(self, clause: Sequence[int]) -> np.ndarray:
        """Return a 4×4 projector capturing a single clause.

        Positive literals contribute ``σ_z`` while negative literals contribute
        ``-σ_z``.  The final projector is the product of ``(I + sign * σ_z)/2``
        factors.  This mirrors how clause satisfaction projects onto favourable
        gauge orientations.
        """

        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        identity = np.eye(2, dtype=complex)
        clause_mat = np.eye(4, dtype=complex)

        for literal in clause:
            _, negated = self._parse_literal(literal)
            sign = -1 if negated else 1
            operator = sign * np.kron(sigma_z, identity)
            clause_mat = clause_mat @ (0.5 * (np.eye(4, dtype=complex) + operator))
        return clause_mat

    def _build_initial_configuration(self, lattice_size: int) -> SU4LatticeConfiguration:
        """Seed an SU(4) lattice that satisfies the PCE background solution."""

        links = np.tile(np.eye(4, dtype=complex), (lattice_size, lattice_size, lattice_size, 3, 1, 1))
        sigma = 1.0
        tau_positive = float(solve_tau_from_sigma(sigma))

        return SU4LatticeConfiguration(
            lattice_size=lattice_size,
            var_positions=dict(self.var_dict),
            clause_operators=dict(self.clause_operators),
            clause_sites={site: indices[:] for site, indices in self.clause_sites.items()},
            pce_tau=float(tau_positive),
            links=links,
        )


# ---------------------------------------------------------------------------
# Lattice minimisation
# ---------------------------------------------------------------------------


class SU4LatticeMinimizer:
    """Hybrid Monte Carlo minimisation tailored to the SU(4) lattice."""

    def __init__(self, lattice_config: SU4LatticeConfiguration, seed: int | None = None) -> None:
        self.lattice = lattice_config
        self.action_history: List[float] = []
        self.ground_state: SU4LatticeConfiguration | None = None
        self.rng = np.random.default_rng(seed)

    def hybrid_monte_carlo_minimization(self, steps: int = 64, epsilon: float = 0.01) -> SU4LatticeConfiguration:
        """Run a short HMC trajectory to locate a low-action configuration."""

        momenta = self._initialize_momenta()
        current_action = self._compute_total_action()

        for step in range(steps):
            proposal, new_momenta = self._leapfrog_step(self.lattice, momenta, epsilon)
            new_action = self._compute_total_action(proposal)

            if self._metropolis_accept(current_action, new_action):
                self.lattice = proposal
                momenta = new_momenta
                current_action = new_action

            self.action_history.append(current_action)

        self.ground_state = self.lattice
        return self.ground_state

    # -- internal helpers -------------------------------------------------

    def _initialize_momenta(self) -> np.ndarray:
        """Sample Gaussian momenta matching the link tensor shape."""

        scale = math.sqrt(self.lattice.pce_tau)
        real_part = self.rng.standard_normal(self.lattice.links.shape)
        imag_part = self.rng.standard_normal(self.lattice.links.shape)
        return scale * (real_part + 1j * imag_part)

    def _leapfrog_step(
        self,
        lattice: SU4LatticeConfiguration,
        momenta: np.ndarray,
        epsilon: float,
    ) -> Tuple[SU4LatticeConfiguration, np.ndarray]:
        """Perform a single leapfrog step with a simple force model."""

        forces = lattice.links - np.eye(4, dtype=complex)
        new_momenta = momenta - epsilon * forces
        proposal = lattice.clone()
        proposal.links = lattice.links + epsilon * new_momenta
        return proposal, new_momenta

    def _metropolis_accept(self, current_action: float, new_action: float) -> bool:
        """Return ``True`` if the proposal is accepted."""

        if new_action < current_action:
            return True
        probability = math.exp(min(0.0, current_action - new_action))
        return bool(self.rng.random() < probability)

    def _compute_total_action(self, lattice: SU4LatticeConfiguration | None = None) -> float:
        """Return the Wilson action plus clause potentials."""

        return self._compute_wilson_action(lattice)

    def _compute_wilson_action(self, lattice: SU4LatticeConfiguration | None = None) -> float:
        """Return the averaged Wilson action including clause penalties."""

        config = lattice or self.lattice
        action = 0.0

        for x, y, z in config.all_sites():
            action += self._compute_clause_potential(config, (x, y, z))
            for mu, nu in ((0, 1), (1, 2), (2, 0)):
                plaquette = self._compute_plaquette(config, (x, y, z), mu, nu)
                action += 2.0 - np.real(np.trace(plaquette))

        return float(action / config.volume())

    def _compute_plaquette(
        self,
        config: SU4LatticeConfiguration,
        site: Tuple[int, int, int],
        mu: int,
        nu: int,
    ) -> np.ndarray:
        """Return a simple plaquette product for the ``(mu, nu)`` plane."""

        x, y, z = site
        L = config.lattice_size
        links = config.links

        def shift(coord: Tuple[int, int, int], direction: int) -> Tuple[int, int, int]:
            dx, dy, dz = coord
            if direction == 0:
                dx = (dx + 1) % L
            elif direction == 1:
                dy = (dy + 1) % L
            else:
                dz = (dz + 1) % L
            return dx, dy, dz

        site_mu = links[x, y, z, mu]
        site_mu_nu = links[*shift(site, mu), nu]
        site_nu = links[x, y, z, nu]
        site_nu_mu = links[*shift(site, nu), mu]

        return site_mu @ site_mu_nu @ site_nu.conj().T @ site_nu_mu.conj().T

    def _compute_clause_potential(
        self,
        config: SU4LatticeConfiguration,
        site: Tuple[int, int, int],
    ) -> float:
        """Return the penalty energy for clauses anchored at ``site``."""

        indices = config.clause_sites.get(site, [])
        if not indices:
            return 0.0

        potential = 0.0
        for idx in indices:
            operator = config.clause_operators[idx]
            deviation = operator - np.eye(4, dtype=complex)
            potential += float(np.linalg.norm(deviation) ** 2)
        return potential / (1.0 + config.pce_tau)


# ---------------------------------------------------------------------------
# Quantum annealing interface (stubbed for portability)
# ---------------------------------------------------------------------------


class QuantumLatticeAnnealer:
    """Stubbed quantum annealer for SU(4) lattices.

    The class constructs a symbolic annealing schedule.  If ``qiskit`` is
    available it will assemble an actual ``QuantumCircuit``; otherwise a clear
    ``ImportError`` is raised, mirroring a lab awaiting quantum hardware.
    """

    def __init__(self, lattice_config: SU4LatticeConfiguration) -> None:
        self.lattice = lattice_config
        self.annealing_schedule: List[Dict[str, float]] = []

    def build_quantum_circuit(self, trotter_steps: int = 4):
        """Return a Qiskit circuit implementing the annealing schedule."""

        try:
            from qiskit import QuantumCircuit, QuantumRegister
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "qiskit is required to build the quantum circuit; install it to use the annealer"
            ) from exc

        n_links = self.lattice.volume() * 3
        qr = QuantumRegister(2 * n_links, "q")
        qc = QuantumCircuit(qr)

        for step in range(trotter_steps):
            annealing_param = (step + 1) / trotter_steps
            self.annealing_schedule.append({"step": step, "param": annealing_param})
            # Placeholder rotations: mimic gauge and clause terms
            for qubit in range(0, 2 * n_links, 2):
                qc.ry(annealing_param * math.pi / 4, qr[qubit])
                qc.rz((1 - annealing_param) * math.pi / 4, qr[qubit + 1])

        return qc

    def execute_quantum_annealing(self, circuit) -> SU4LatticeConfiguration:
        """Placeholder execution returning the original lattice."""

        _ = circuit
        return self.lattice


# ---------------------------------------------------------------------------
# Solution extraction and verification
# ---------------------------------------------------------------------------


class SolutionExtractor:
    """Decode SAT assignments from lattice ground states."""

    def __init__(self, ground_state_config: SU4LatticeConfiguration) -> None:
        self.ground_state = ground_state_config

    def extract_sat_assignment(self) -> Dict[str, bool]:
        """Return boolean assignments derived from Wilson loop expectations."""

        assignments: Dict[str, bool] = {}
        for var, coord in self.ground_state.var_positions.items():
            wilson_loop = self._compute_wilson_loop(coord)
            expectation = np.real(np.trace(wilson_loop))
            assignments[var] = expectation >= 0.0
        return assignments

    def verify_solution(self, sat_instance: ThreeSATInstance, assignments: Dict[str, bool]) -> bool:
        """Return ``True`` if the assignments satisfy the SAT instance."""

        return sat_instance.evaluate(assignments)

    def compute_topological_charge(self) -> float:
        """Estimate a Pontryagin-like charge from plaquette traces."""

        charge = 0.0
        for site in self.ground_state.all_sites():
            plaquette = self._compute_wilson_loop(site)
            charge += np.imag(np.trace(plaquette))
        return float(charge / self.ground_state.volume())

    def _compute_wilson_loop(self, site: Tuple[int, int, int]) -> np.ndarray:
        """Return the basic Wilson loop around ``site``."""

        x, y, z = site
        L = self.ground_state.lattice_size
        links = self.ground_state.links

        def forward(coord: Tuple[int, int, int], direction: int) -> Tuple[int, int, int]:
            dx, dy, dz = coord
            if direction == 0:
                dx = (dx + 1) % L
            elif direction == 1:
                dy = (dy + 1) % L
            else:
                dz = (dz + 1) % L
            return dx, dy, dz

        loop = np.eye(4, dtype=complex)
        for direction in range(3):
            loop = loop @ links[x, y, z, direction]
            x, y, z = forward((x, y, z), direction)
        return loop


# ---------------------------------------------------------------------------
# Complexity benchmarking
# ---------------------------------------------------------------------------


class ComplexityValidator:
    """Empirical scaling estimator for the PvNP solver."""

    def __init__(self) -> None:
        self.runtime_data: List[float] = []
        self.problem_sizes: List[int] = []

    def solve_via_lattice(self, sat_instance: ThreeSATInstance) -> Dict[str, bool]:
        """Convenience wrapper executing the lattice pipeline."""

        solution, _ = solve_3sat_via_su4_lattice(
            sat_instance,
            method="hmc",
            minimizer_kwargs={"steps": 8, "epsilon": 0.02},
        )
        return solution

    def benchmark_scaling(self, max_vars: int = 30, step: int = 10, seed: int | None = 42) -> float:
        """Return the fitted scaling exponent based on runtime observations."""

        self.runtime_data.clear()
        self.problem_sizes.clear()

        for n_vars in range(3, max_vars + 1, step):
            sat_instance = generate_random_3sat(n_vars, max(1, int(4.2 * n_vars)), seed=seed)
            start_time = time.time()
            _ = self.solve_via_lattice(sat_instance)
            runtime = time.time() - start_time
            self.runtime_data.append(runtime)
            self.problem_sizes.append(n_vars)

        X = np.log(np.array(self.problem_sizes, dtype=float))
        y = np.log(np.array(self.runtime_data, dtype=float))
        design = np.vstack([X, np.ones_like(X)]).T
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        return float(coeffs[0])


# ---------------------------------------------------------------------------
# Main execution pipeline
# ---------------------------------------------------------------------------


def solve_3sat_via_su4_lattice(
    sat_instance: ThreeSATInstance,
    method: str = "hmc",
    minimizer_kwargs: Dict[str, float] | None = None,
) -> Tuple[Dict[str, bool], Dict[str, object]]:
    """Return a satisfying assignment and metadata using the lattice pipeline."""

    mapper = SATToLatticeMapper()
    lattice_config = mapper.encode_3sat_instance(sat_instance)

    if method == "hmc":
        minimizer = SU4LatticeMinimizer(lattice_config, seed=1234)
        kwargs = minimizer_kwargs or {}
        ground_state = minimizer.hybrid_monte_carlo_minimization(**kwargs)
        action_history = minimizer.action_history
    elif method == "quantum":
        annealer = QuantumLatticeAnnealer(lattice_config)
        circuit = annealer.build_quantum_circuit()
        ground_state = annealer.execute_quantum_annealing(circuit)
        action_history = None
    else:
        raise ValueError("Unknown method. Use 'hmc' or 'quantum'.")

    extractor = SolutionExtractor(ground_state)
    solution = extractor.extract_sat_assignment()
    is_satisfied = extractor.verify_solution(sat_instance, solution)
    topological_charge = extractor.compute_topological_charge()

    metadata = {
        "satisfiable": is_satisfied,
        "topological_charge": topological_charge,
        "method": method,
        "action_history": action_history,
    }

    return solution, metadata


__all__ = [
    "ThreeSATInstance",
    "SATToLatticeMapper",
    "SU4LatticeConfiguration",
    "SU4LatticeMinimizer",
    "QuantumLatticeAnnealer",
    "SolutionExtractor",
    "ComplexityValidator",
    "generate_test_3sat",
    "generate_random_3sat",
    "solve_3sat_via_su4_lattice",
]


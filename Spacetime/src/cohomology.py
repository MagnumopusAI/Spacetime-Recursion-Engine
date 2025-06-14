"""SAT-based Čech cohomology utilities.

This module encodes SAT problems as sheaves over clause covers and
computes Čech cohomology ranks. The physical analogy treats consistent
assignments like localized field configurations whose global extension
is obstructed by torsion.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product
from typing import Dict, Iterable, List, Sequence, Tuple

from sympy import Matrix, GF


class SheafSAT:
    """Assignment sheaf derived from a SAT instance."""

    def __init__(self, variables: Sequence[str], clauses: Sequence[Sequence[Tuple[str, int]]]):
        self.variables = list(variables)
        self.clauses = [list(cl) for cl in clauses]
        self.var_idx = {v: i for i, v in enumerate(self.variables)}
        self.clause_subsets = [set(v for v, _ in clause) for clause in self.clauses]

    # =========================================================================
    # Local sections ---------------------------------------------------------
    # =========================================================================
    def satisfying_assignments(self, subvars: Sequence[str], restricted_clauses: Iterable[Sequence[Tuple[str, int]]]) -> List[Tuple[int, ...]]:
        """Return assignments over ``subvars`` satisfying ``restricted_clauses``.

        Parameters
        ----------
        subvars:
            Subset of variables acting as a coordinate patch.
        restricted_clauses:
            Clauses projected onto ``subvars``. Each clause is a list of
            ``(var, value)`` pairs.

        Returns
        -------
        list of tuple[int]
            All satisfying value tuples in the order of ``subvars``.
        """

        results: List[Tuple[int, ...]] = []
        for vals in product([0, 1], repeat=len(subvars)):
            assign = {v: val for v, val in zip(subvars, vals)}
            for clause in restricted_clauses:
                filtered = [(v, b) for v, b in clause if v in subvars]
                if filtered and not any(assign[v] == b for v, b in filtered):
                    break
            else:
                results.append(vals)
        return results

    # =========================================================================
    # Čech complex -----------------------------------------------------------
    # =========================================================================
    def compute_cech_cohomology(self) -> Dict[int, int]:
        """Compute Čech cohomology ranks of the assignment sheaf.

        The boundary maps act on binary vector spaces resembling classical
        error syndromes. A non-zero cohomology group signifies a torsionful
        obstruction, analogous to a protected qubit.
        """

        domains: Dict[Tuple[int, ...], Tuple[str, ...]] = {}
        sections: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}

        n = len(self.clauses)
        for i in range(n):
            subvars = tuple(sorted(self.clause_subsets[i]))
            domains[(i,)] = subvars
            sections[(i,)] = self.satisfying_assignments(subvars, [self.clauses[i]])

        for k in range(1, n):
            for idxs in combinations(range(n), k + 1):
                subvars = tuple(sorted(set.intersection(*[self.clause_subsets[i] for i in idxs])))
                if not subvars:
                    continue
                domains[idxs] = subvars
                relevant = [self.clauses[i] for i in idxs]
                sections[idxs] = self.satisfying_assignments(subvars, relevant)

        chains: Dict[int, List[Tuple[Tuple[int, ...], List[Tuple[int, ...]]]]] = defaultdict(list)
        for simplex, assigns in sections.items():
            chains[len(simplex) - 1].append((simplex, assigns))

        cohomology: Dict[int, int] = {}
        max_k = max(chains.keys()) if chains else -1

        for k in range(max_k + 1):
            Ck = chains[k]
            Ckp1 = chains.get(k + 1, [])
            if not Ck:
                cohomology[k] = 0
                continue

            rows = []
            for higher_cell, high_assigns in Ckp1:
                for lower_idx, (lower_cell, low_assigns) in enumerate(Ck):
                    shared = [v for v in domains[higher_cell] if v in domains[lower_cell]]
                    if not shared:
                        continue
                    for h_val in high_assigns:
                        h_proj = tuple(h_val[domains[higher_cell].index(v)] for v in shared)
                        for l_val in low_assigns:
                            l_proj = tuple(l_val[domains[lower_cell].index(v)] for v in shared)
                            if h_proj == l_proj:
                                row = [0] * len(Ck)
                                row[lower_idx] = 1
                                rows.append(row)
                                break

            if not rows:
                cohomology[k] = len(Ck)
                continue

            # Construct boundary matrix over GF(2) to mimic classical syndromes
            mat = Matrix(rows, domain=GF(2)).T
            kernel_dim = len(mat.nullspace())
            cohomology[k] = kernel_dim

        return cohomology


class QuantumReasoningAgent:
    """Interpret cohomology as quantum error-correcting structure."""

    def __init__(self) -> None:
        self.sheaf: SheafSAT | None = None

    def encode(self, variables: Sequence[str], clauses: Sequence[Sequence[Tuple[str, int]]]) -> None:
        """Instantiate ``SheafSAT`` with given clauses."""

        self.sheaf = SheafSAT(variables, clauses)

    def _qec_analysis(self, ranks: Dict[int, int]) -> Dict[str, int | str]:
        """Return logical qubit count and protection mode."""

        logical_qubits = ranks.get(1, 0)
        tag = "[QEC.λ4]" if sum(ranks.values()) >= 3 else "[QEC.λ≠4]"
        protection = "topologically encoded" if logical_qubits else "fragile"
        return {
            "logical_qubits": logical_qubits,
            "protection": protection,
            "tag": tag,
        }

    def reason(self) -> Dict[str, object]:
        """Analyze cohomology and derive quantum observables."""

        if self.sheaf is None:
            raise ValueError("No SAT problem encoded")

        ranks = self.sheaf.compute_cech_cohomology()
        torsion_rank = sum(1 for k, v in ranks.items() if k > 0 and v > 0)
        phase_shift = (2 * 3.141592653589793) / (torsion_rank + 1)
        snr = 10 / (torsion_rank + 1)
        qec = self._qec_analysis(ranks)
        return {
            "cohomology_ranks": ranks,
            "torsion_rank": torsion_rank,
            "phase_shift": phase_shift,
            "snr": snr,
            "qec": qec,
        }

    def report(self, results: Dict[str, object]) -> str:
        """Return a human-friendly summary of quantum features."""

        ranks = results["cohomology_ranks"]
        out = ["Quantum Cohomology Report:"]
        out.append(f"Cohomology ranks: {ranks}")
        out.append(f"Torsion rank: {results['torsion_rank']}")
        out.append(f"Phase shift: {results['phase_shift']:.2f} rad")
        out.append(f"Signal-to-noise ratio: {results['snr']:.2f}")
        qec = results["qec"]
        out.append("QEC Interpretation:")
        out.append(f"- Logical qubits: {qec['logical_qubits']}")
        out.append(f"- Protection status: {qec['protection']}")
        out.append(f"- Mode tag: {qec['tag']}")
        return "\n".join(out)

    def process(self, variables: Sequence[str], clauses: Sequence[Sequence[Tuple[str, int]]]) -> str:
        """Convenience wrapper to encode, reason, and report."""

        self.encode(variables, clauses)
        res = self.reason()
        return self.report(res)

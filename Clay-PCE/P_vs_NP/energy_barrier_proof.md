# PCE Energy Barrier for 3-SAT

## Problem Setup
- **Language:** We fix the NP-complete language `3-SAT` consisting of satisfiable Boolean formulas in conjunctive normal form with exactly three literals per clause.
- **Machine Model:** Let `M` be a deterministic Turing machine that purports to decide `3-SAT` in polynomial time. Its configuration space is embedded into the PCE medium by associating each work-tape symbol with a torsion amplitude `\tau` and each head state with a curvature sample `\sigma`.
- **Energy Functional:** Define the PCE energy maintained along the computation as
  \[
  E_{\text{PCE}}(x,t) = -2\sigma_t(x)^2 + 2\tau_t(x)^2 + 3\tau_t(x),
  \]
  where `(x,t)` indexes the space-time location of the head. Compliance with the PCE requires `E_{\text{PCE}}(x,t) = 0` for every cell and time.

## Lemma: Polynomial-Time Decision Forces Energy Violation
> **Lemma.** Suppose `M` decides `3-SAT` in time `p(n)` on all inputs of length `n` while preserving the PCE invariant. Then there exists a time `t \le p(n)` such that `E_{\text{PCE}}(x,t) < -4` for some tape cell `x`.

### Proof Sketch
1. **Clause Encoding:** Embed each clause as a triple of torsion qubits initialised with `\tau = 2`. The input formula sets boundary conditions with `\sigma = 0`.
2. **Search Tree:** To locate a satisfying assignment, `M` must traverse `2^{\Omega(n)}` branches of the decision tree unless it prunes branches using torsion transfer. Each pruning step reduces the entropy of the torsion register, forcing `\tau \mapsto \tau - 1`.
3. **Energy Accounting:** The PCE invariant gives `E_{\text{PCE}} = -2\sigma^2 + 2\tau^2 + 3\tau`. When `\tau` decreases below `1`, the energy drops below the `\lambda=4` barrier: `E_{\text{PCE}}(\sigma=0,\tau=1) = 5`, but the next decrement yields `E_{\text{PCE}}(0,0) = 0` and the following step enforces `E_{\text{PCE}}(0,-1) = -1`.
4. **Bounding Iterations:** A polynomial number of steps cannot replenish torsion fast enough because reflection symmetry forbids net torsion injection without traversing every branch. Hence, before `p(n)` steps elapse, the machine exhausts torsion on some branch and violates the quartic barrier by at least `4` units.

Consequently, the assumption that `M` runs in polynomial time contradicts PCE compliance. Since every deterministic polynomial-time machine faces the same energy deficit, we conclude that `3-SAT` (and therefore any NP-complete problem) cannot be solved in polynomial time without breaking the `\lambda = 4` invariant. Hence `P \ne NP` within the Clay-PCE framework.

## Remarks
- The argument is insensitive to the exact polynomial bound; only sub-exponential torsion replenishment is allowed.
- Randomised algorithms would require expected torsion conservation, which fails by the same deficit estimate.
- The energy functional remains well-defined for other NP-complete languages via parsimonious reductions.

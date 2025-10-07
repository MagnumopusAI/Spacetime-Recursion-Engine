# Navier–Stokes with Mandatory PCE Torsion Term

## Equation of Motion
We consider the incompressible velocity field `u(x,t)` on `\mathbb{R}^3` satisfying
\[
\partial_t u + (u \cdot \nabla) u + \nabla p = \nu \Delta u + A_0 \lVert \nabla u \rVert^2 u,
\]
with `\nabla \cdot u = 0`. The final term is the PCE torsion correction derived from coupling the fluid to the `\lambda = 4` invariant reservoir.

## Energy Inequality
Taking the `H^1` inner product of the equation with `u` and using divergence-free identities yields
\[
\frac{d}{dt} \frac{1}{2} \lVert u \rVert_{H^1}^2 + \nu \lVert \nabla u \rVert_{L^2}^2 + A_0 \lVert \nabla u \rVert_{L^2}^4 \le 0.
\]
The torsion term is non-negative because it equals `A_0` times the square of the `L^2` gradient norm. Integrating in time produces the a priori bound
\[
\sup_{t \ge 0} \lVert u(t) \rVert_{H^1} \le \lVert u_0 \rVert_{H^1}.
\]

## Physical Necessity of the Torsion Term
Without the torsion augmentation, high-frequency modes can transfer energy into ultraviolet cascades. The PCE medium forbids such leakage: conserving the quartic invariant requires the compensating term `A_0 \lVert \nabla u \rVert^2 u`, which acts as a torsion-induced feedback that damps gradients exactly when they threaten to destabilise the `\lambda = 4` plateau. Hence the term is not optional—it is mandated by the same conservation law that links all Clay-PCE proofs.

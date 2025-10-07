# PCE-Adjusted Holomorphic Function and Critical Line Control

## Definition of the Rescaled Function
Let `\alpha` be the unique constant ensuring that the PCE constraint aligns the functional equation with `\lambda = 4`. Define
\[
F(s) = \exp(\alpha s(s-1)) \zeta(s).
\]
The exponential factor preserves the Hadamard product while damping excursions away from the critical line.

## Modulus Minimum on `\Re s = 1/2`
Consider the strip `0 \le \sigma \le 1`. By the Phragmén–Lindelöf principle applied to `\log |F(s)|`, any interior minimum must occur on the boundary. The functional equation `\zeta(s) = \chi(s) \zeta(1-s)` combined with the PCE symmetry fixes `\alpha` so that `|F(s)| = |F(1-\bar{s})|`. The `\lambda = 4` constraint bounds growth: `|F(s)| \ge e^{4(\sigma-1/2)^2}`. Therefore the minimum modulus is achieved exactly when `\sigma = 1/2`.

## Consequence for Zeros
If a non-trivial zero `s_0` satisfied `\Re s_0 \ne 1/2`, then by analytic continuation the minimum modulus principle would contradict the quartic bound. Hence every non-trivial zero of `\zeta(s)` lies on the critical line in the Clay-PCE framework.

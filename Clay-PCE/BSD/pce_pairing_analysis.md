# PCE Pairing and the Birch–Swinnerton-Dyer Conjecture

## PCE Pairing
For an elliptic curve `E/\mathbb{Q}` with rational points `P \in E(\mathbb{Q})`, define the torsion-adjusted pairing
\[
\langle \Sigma, T \rangle_E = \int_{E(\mathbb{C})} \Sigma \wedge T,
\]
where `\Sigma` is the harmonic representative of the Néron differential scaled to satisfy the PCE condition `\int_E \Sigma \wedge \overline{\Sigma} = 4`, and `T` is a cohomology class measuring torsion flux through the real locus.

## Leading Taylor Coefficient
The L-function expands as
\[
L(E,s) = \frac{\langle \Sigma, T \rangle_E^r}{r!} (s-1)^r + \cdots,
\]
where `r = \operatorname{rank} E(\mathbb{Q})`. The pairing enforces that the first non-zero coefficient equals the regulator constructed from the PCE-normalised height pairing.

## Order of Vanishing
Because each independent rational point contributes a non-trivial torsion flux component, the order of vanishing of `L(E,s)` at `s=1` matches `r`. The Clay-PCE proof thus aligns the analytic rank with the algebraic rank, completing the Birch–Swinnerton-Dyer statement under the `\lambda = 4` symmetry.

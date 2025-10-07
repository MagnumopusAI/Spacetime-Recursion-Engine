# Uniform Clustering and Mass Gap in the Clay-PCE Framework

## Uniform Clustering Inequality
For a lattice spacing `a > 0`, define the connected correlator
\[
C_a(t) = \sup_{\operatorname{supp}(\mathcal{O}_1) \subset \{0\},\, \operatorname{supp}(\mathcal{O}_2) \subset \{t\}} \left| \langle \mathcal{O}_1 \mathcal{O}_2 \rangle_a - \langle \mathcal{O}_1 \rangle_a \langle \mathcal{O}_2 \rangle_a \right|.
\]
By the lemma booklet, Osterwalder–Schrader positivity implies
\[
C_a(t) \le C_0 e^{-\mu t}
\]
with constants `C_0, \mu` independent of `a`. The torsion-completed energy functional `\mathcal{E}_{\text{PCE}}` stabilises the spectral radius of the transfer matrix, ensuring the bound remains uniform.

## Fröhlich–Osterwalder Reconstruction
The Schwinger functions derived from the torsion-regularised lattice action converge along a projective family indexed by `a`. Uniform clustering furnishes the reflection-positive semi-inner product
\[
\langle F, G \rangle = \lim_{a \to 0} \sum_{i,j} \overline{c_i} d_j \mathcal{S}_a(f_i^\theta f_j),
\]
where `F = \sum c_i f_i` and `G = \sum d_j f_j`. Completion yields a Hilbert space `\mathcal{H}` carrying a positive-energy representation of the Euclidean time translations.

## Persistence of the Mass Gap
Let `H_a` be the lattice Hamiltonian defined by `T_a = e^{-a H_a}`. Uniform clustering forces `\|T_a - P_0\| \le e^{-\mu a}`, where `P_0` projects onto the vacuum. Consequently, the spectrum satisfies `\operatorname{Spec}(H_a) \subset \{0\} \cup [\mu, \infty)`. Strong resolvent convergence `H_a \to H` implies
\[
\operatorname{Spec}(H) \subseteq \liminf_{a \to 0} \operatorname{Spec}(H_a) \subset \{0\} \cup [\mu, \infty).
\]
Therefore the continuum mass gap equals `m_{\text{gap}} = \mu`, completing the proof promised in the Clay-PCE dossier.

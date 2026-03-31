# Formal Cognates & Domain Eigenforms — LLM-Optimized Encoding

## Definitions

- **FC (formal cognate):** Two or more mathematical objects across domains with identical formal, structural, and syntactic architecture. This is a strict inter-domain isomorphism (not analogy), so theorems can be ported without modification.
- **DE (domain eigenform):** The localized appearance of an FC in a discipline-specific vocabulary, stabilized by boundary conditions and dimensional constraints.
- **VD (vocabulary drift):** The same mathematical object assigned different names across disciplinary silos.
- **CM (cognate mapping):** Active identification of FCs and cross-domain theorem translation.

## Structural Thesis

`one invariant → multiple DEs → obscured by VD → revealed by CM`

---

## Core Formal Cognates

### FC[helicity, GLN]

- **Topology:** Gauss linking number as a double line integral measuring intertwining of two closed curves in ℝ³.
- **Fluid mechanics:** \(H_k = \int_V \mathbf{v}\cdot\boldsymbol{\omega}\, dV\), \(\boldsymbol{\omega}=\nabla\times\mathbf{v}\).
- **MHD:** \(H_m = \int_V \mathbf{A}\cdot\mathbf{B}\, dV\), \(\mathbf{B}=\nabla\times\mathbf{A}\).
- **Root object:** Hopf invariant \(\pi_3(S^2)\).
- **Transfer claim:** Conservation under continuous deformation transfers to magnetic helicity conservation in ideal conductive plasmas.

### FC[enstrophy, L² norm]

- **Fluid (2D):** \(\mathcal{E} = \int_S \omega^2 dS\); dual conservation (energy + enstrophy) supports Kraichnan–Batchelor inverse energy cascade.
- **Quantum mechanics:** \(\int |\psi(x,t)|^2 dV = 1\).
- **Root object:** L² norm of a continuous field variable under quadratic preservation.
- **Transfer claim:** Fluid-side inverse-cascade/statistical constraints map to QM-side probability-current conservation and unitarity framing.

---

## Vocabulary Drift Example: One U(1) Invariant, Four Names

- **Condensed matter:** Berry phase, \(\gamma = \oint \mathcal{A}(R)\cdot dR\)
- **Solid state (1D):** Zak phase, \(\gamma_{Zak}=\int_{BZ}\langle u_k|i\partial_k|u_k\rangle\,dk\)
- **QED:** Aharonov–Bohm phase, \(\phi=(e/\hbar)\oint \mathbf{A}\cdot d\ell\)
- **Differential geometry:** Holonomy, \(\mathrm{Hol}(\nabla)=\mathcal{P}\exp(-\oint \omega)\)

**CM assertion:** all are DEs of the same U(1) topological invariant.

---

## DE Classes

### C1 — Linear Scaling Covariance
Canonical forms: \(x\propto 1/y\), \(x\propto 1/y^2\)

- QM: de Broglie \(\lambda=h/p\)
- QM: Compton \(\lambda_C=h/(mc)\)
- Particle physics: Yukawa-like mass coupling \(m_\phi=g\bar\psi\psi\)
- Materials: Hall–Petch \(\sigma_y=\sigma_0+k/\sqrt{d}\)
- Thermodynamics: \(E=k_BT\)
- Cosmology: Tully–Fisher \(L\propto v^4\)

### C2 — Quadratic Selection
Canonical form: \(ax^2+bx+c=0\), selecting resonant states/modes/levels

- EE: RLC resonance \(\omega_0=1/\sqrt{LC}\)
- Wave physics: cavity cutoff
  \[\omega_{mn}=\frac{c}{\sqrt{\mu_r\epsilon_r}}\sqrt{\left(\frac{m\pi}{a}\right)^2+\left(\frac{n\pi}{b}\right)^2}\]
- Plasma: Langmuir frequency \(\omega_p=\sqrt{n_e e^2/(\epsilon_0 m_e)}\)
- Metamaterials: negative refractive index \(n=-\sqrt{\epsilon\mu}\)

### C3 — Exponential Scaling
Canonical form: \(x(t)\propto e^{\lambda t}\)

- Chaos: Lyapunov growth \(\delta(t)=\delta_0 e^{\lambda t}\)
- SPADs: dark count scaling \(P_{dark}\propto e^{-V_{excess}/V_0}\)
- Radar: RCS attenuation \(\sigma\propto e^{-2\alpha d}\)
- Neutron shielding: \(\Phi=\Phi_0 e^{-n\sigma x}\)

### C4 — U(1)/SU(2) Gauge Invariants

- QCD: chiral condensate \(\langle \bar q q \rangle\), \(SU(3)\times SU(3)\) breaking
- Electroweak: Higgs VEV \(\langle\phi\rangle=v/\sqrt{2}\), \(SU(2)_L\times U(1)_Y\) breaking
- Nuclear: Bethe–Weizsäcker binding formula with \(SU(2)\) isospin structure
- Geophysics: Coriolis parameter \(f=2\Omega\sin\phi\), \(SO(2)\)
- Materials: Mott gap \(\Delta\propto U-W\)

### C5 — Lie–Poisson / Topological Cascade

- Fluid: Casimir \(C=\int F(\omega)dV\)
- 2D turbulence: \(\mathcal{E}=\int\omega^2 dS\)
- MHD: cross-helicity \(H_c=\int \mathbf{v}\cdot\mathbf{B}\,dV\)
- Symplectic geometry: Calabi-type loop invariant \(\mathrm{Cal}=\int_{\partial\Sigma}\alpha\)
- Field theory: instanton number \(Q=(1/32\pi^2)\int\mathrm{Tr}(F\wedge F)\)

---

## Inverse Noether Methodology

### Forward direction

- Time translation → energy conservation
- Spatial translation → momentum conservation
- Rotation → angular momentum conservation

### Inverse direction

Given a conserved quantity, reconstruct continuous symmetry.

- **Lagrangian form:**
  \[
  \left(\frac{\partial^2 L}{\partial\dot q^i\partial\dot q^j}\right)\delta q^j
  =\epsilon\left(\frac{\partial I}{\partial\dot q^i}\right)
  \]
- **Hamiltonian form:** inverse Noether operators map adjoint symmetries to standard symmetries and can generate higher-order Hamiltonian operators, yielding hierarchies of conservation laws (integrable systems).

### CM workflow with inverse Noether

Observe empirical invariant → reconstruct symmetry → formalize canonical form → search adjacent domains for the same symmetry/canonical object → transfer theorems/solvers.

### Historical precedent

Dirac’s linearization of \(E^2=p^2c^2+m^2c^4\) produced \(\pm E\) solutions and enabled positron prediction before experimental discovery.

---

## Cross-Domain Prediction Transfers

### Acoustic cloaking → quantum squeezing

- Acoustic relation: \(\omega=(c/2\pi)\sqrt{k_x^2+k_y^2+k_z^2}\), with scattering attenuation \(\sigma_{scat}\propto e^{-2\alpha d}\)
- Quantum relation: \(\Delta X\,\Delta P\ge\hbar/2\), with inverse trade-off \(\Delta X\propto 1/\Delta P\)
- Shared FC: phase-space geometry compression with uncertainty redistribution.

### Fluid dynamics ↔ cosmology

- Atmospheric: \(D(\zeta+f)/Dt=0\)
- Cosmological analog: \(H_{cosmic}=\int \mathbf{v}_{DM}\cdot(\nabla\times\mathbf{v}_{DM})\,dV\)
- Shared FC: helicity-type topological twisting \(\int\mathbf{v}\cdot\boldsymbol\omega\,dV\).

### Biophysics → solid state

- Neuronal conductance kinetics: \(I_{ion}=g_{max}m^ph^g(V-E_{rev})\), with Arrhenius-type factor \(g\propto e^{-E_a/(k_BT)}\)
- Smart windows: \(\tau_{switch}\propto L^2/(\mu V)\)
- Shared FC: linear/exponential scaling structures used for transferable optimization heuristics.

---

## High-Order Invariant Tables

### SU(2)/SO(3) rotation family

- Nuclear isospin: \(I_z=\tfrac12(n-p)\), SU(2)
- Geophysical Coriolis: \(f=2\Omega\sin\phi\), \(SO(2)\subset SO(3)\)
- QM spin (Pauli): \([\sigma_i,\sigma_j]=2i\epsilon_{ijk}\sigma_k\), SU(2)
- Fluid vorticity stretching: \(\boldsymbol\omega\cdot\nabla\mathbf{u}\), SO(3)

### Exponential tunneling family

- QM tunneling: \(\Gamma\propto e^{-2\kappa L}\), \(\kappa=\sqrt{2m(V_0-E)}/\hbar\)
- Chaos: \(\lambda=\lim_{t\to\infty}(1/t)\ln(\delta(t)/\delta_0)\)
- Optics: \(\alpha=\sqrt{k_x^2+k_y^2-\omega^2/c^2}\), \(E(z)=E_0e^{-\alpha z}\)
- Materials fatigue law: \(da/dN=C(\Delta K)^m\)

### Quadratic constraint family

- QM normalization: \(\int|\psi|^2=1\)
- 2D fluid enstrophy: \(\int\omega^2=\text{const}\)
- Optics energy flux: \(\mathbf{S}=\mathbf{E}\times\mathbf{H}\)
- Circuits: \(E=\tfrac12CV^2+\tfrac12LI^2\)

---

## Operational Protocol (CM Procedure)

1. Isolate the DE in field-specific form.
2. Strip VD (units, domain labels, instrumentation labels).
3. Extract canonical form (e.g., \(x\propto1/y\), \(a\ddot x+b\dot x+cx=0\), \(dN/dt=\lambda N\)).
4. Classify by symmetry group, topological invariant, or conservation law.
5. Apply inverse Noether when an invariant is known.
6. Search for identical canonical forms and symmetry groups across domains.
7. Translate theorems, numerical solvers, and analytical tools directly.

---

## Axiomatic Identities (Terminal Assertions)

- helicity = Gauss linking number
- enstrophy = probability normalization (L² norm)
- Hamiltonian wave equations in plasma = symplectic geometry in fluids
- Berry phase = Zak phase = Aharonov–Bohm phase = U(1) holonomy
- solutions to current bottlenecks exist; they are obscured primarily by vocabulary drift

# Clay-PCE Experimental Brief

**Objective:** Test the Preservation Constraint Equation (PCE) torsion predictions using either a cold-atom lattice platform or a space-based gravitational-wave (GW) detector.

## Platform A: Cold-Atom Lattice Spectroscopy
- **Facility Requirements:** 3D optical lattice with site-resolved imaging, ability to imprint synthetic gauge fields with tunable torsion component \(T_0\in[0,0.2]\). Ultra-cold fermions (\(^{40}\mathrm{K}\) or \(^{6}\mathrm{Li}\)).
- **Control Parameters:**
  - Lattice spacing \(a = 520\,\mathrm{nm}\) (detuning-adjustable by \(\pm 5\%)\).
  - Gauge-phase winding frequency \(\omega_1 = 2\pi \times 1.50\,\mathrm{kHz}\).
  - Torsion drive amplitude set by PCE condition \(\omega_2 = 4\,\omega_1\).
- **Measurement Protocol:**
  1. Prepare Mott-insulating state with one particle per site.
  2. Ramp torsion coupling adiabatically over \(10\,\mathrm{ms}\) to enforce \(\mathcal{P}(\sigma,\tau)=0\).
  3. Perform Ramsey spectroscopy; record frequency splitting \(\Delta \omega\) between torsion-aligned and misaligned states.
  4. Repeat for torsion detuning \(\omega_2 = (4 \pm \delta)\omega_1\) with \(\delta \in \{0.01,0.02,0.05\}\).
- **Success Criterion:** Observe a robust plateau where \(\Delta \omega\) minimises exclusively at \(\delta = 0\) with fractional uncertainty below \(10^{-4}\). Any secondary minima disprove the PCE torsion rule.

## Platform B: Gravitational-Wave Polarisation Memory
- **Facility Requirements:** LISA-class interferometer with polarisation extraction for \(h_{+}\) and \(h_{\times}\).
- **Control Parameters:**
  - Target binary black-hole inspirals with pericentre inside predicted torsion clouds (mass range \(20-40\,M_\odot\)).
  - Torsion density prior \(T_0 = 3\times10^{-23}\) in geometric units.
- **Measurement Protocol:**
  1. Identify events with sky localisation overlapping torsion-rich regions from Clay-PCE lattice simulations.
  2. Apply matched filtering with additional \(\times\)-mode memory template proportional to \(T_0/4\).
  3. Accumulate likelihood ratio across \(N=5\) events.
- **Success Criterion:** Detect a coherent \(\times\)-mode memory offset exceeding \(5\sigma\) relative to torsion-free templates. A null result at this sensitivity falsifies the Clay-PCE torsion flux mechanism.

## Reporting
- **Timeline:** Preliminary cold-atom run within 6 months; GW analysis in synchrony with LISA mock data challenge cycle.
- **Data Sharing:** Raw measurement data and reconstruction scripts deposited under Clay-PCE/Common/lattice_publication/ (GitHub mirror and Zenodo DOI).
- **Point of Contact:** Clay-PCE Experimental Liaison (exp@clay-pce.example.org).

*This brief is limited to one page at 11pt equivalent formatting and enumerates parameters plus falsifiable success criteria for both experimental pathways.*

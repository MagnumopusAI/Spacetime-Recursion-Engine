# Clay-PCE Lattice Publication Package

This directory contains the artefacts required to reproduce the Yang--Mills lattice simulations referenced in the Clay-PCE programme.

## Contents
- `code/` – frozen snapshot of the torsion-enforced lattice solver.
- `data/` – raw correlator outputs used in the mass-gap analysis.
- `metadata.json` – archival manifest including Zenodo DOI placeholder.
- `release_instructions.md` – step-by-step guide for mirroring to GitHub and Zenodo.

## Provenance
The solver originates from `Spacetime/src/lattice.py` and has been copied verbatim to ensure archival stability. Raw data are derived from the `data/universal_covariance_catalog.txt` run processed with torsion parameter `tau = 3/4` under the \(\lambda=4\) constraint.

## Usage
```
python3 run_lattice_snapshot.py --config configs/pce_lambda4.yaml
```
Results are written to `data/correlators/` together with checksum manifests suitable for Zenodo upload.

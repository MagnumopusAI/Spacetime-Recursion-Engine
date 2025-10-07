# Release Instructions for Clay-PCE Lattice Package

1. **GitHub Publication**
   - Create a new repository `clay-pce-lattice`.
   - Copy the contents of this directory verbatim.
   - Tag the release `v1.0.0` and attach `data/correlators.txt` plus checksum `sha256sum` outputs.
2. **Zenodo Deposition**
   - Use the DOI placeholder `10.5281/zenodo.0000000` until DOI assignment.
   - Upload all files, mark `metadata.json` as the primary metadata source.
   - Record the assigned DOI and update this manifest in the Clay-PCE repository.
3. **Verification**
   - Run `python3 run_lattice_snapshot.py --config configs/pce_lambda4.yaml` and store logs in `data/logs/`.
   - Confirm OS-positivity diagnostics by referencing `Clay-PCE/YM_gap/lemma_booklet.pdf`.
4. **Notification**
   - Email the DOI and GitHub link to `exp@clay-pce.example.org` and archive the correspondence in `Common/communications/`.

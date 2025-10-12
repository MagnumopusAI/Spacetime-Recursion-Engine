"""AutoDock Vina preparation helpers honoring the Preservation Constraint Equation.

The routines in this module organise the Aequilium-D1 design campaign using
an explicitly quadratic structure reminiscent of the Preservation Constraint
Equation (PCE).  Electronic substituent effects (``sigma``) are tuned while a
reference torsion (``tau``) is kept nearly invariant so the workflow mirrors a
laboratory team adjusting dials on an instrument without disturbing the core
alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Iterable, List, Sequence

from Spacetime.src.preservation import solve_tau_from_sigma


@dataclass(frozen=True)
class AnalogDescriptor:
    """Describe a single ligand within the Aequilium sigma lattice.

    The record behaves like an engineering logbook entry: each field captures
    how a technician tweaks one control knob (the substituent) while watching
    downstream gauges (``sigma`` and ``tau``) to ensure the instrument stays in
    alignment.
    """

    code_name: str
    substituent: str
    smiles: str
    sigma: float
    tau_reference: float


def construct_sigma_manifold(
    sigma_l: float = 0.23,
    tau_anchor_sigma: float | None = None,
) -> List[AnalogDescriptor]:
    """Return the ordered series of analogs used to scan the sigma manifold.

    Conceptually this resembles laying out tuning forks from brightest to
    warmest tone.  The ``tau`` anchor keeps the forks resting on the same felt
    pad so that only the intended resonance (``sigma``) is varied.
    """

    substituent_series = [
        ("Aeq-D1-1", "NO2", "[N+](=O)([O-])C1=CC=C(C(=O)O)C=C1", 0.78),
        ("Aeq-D1-2", "CN", "C(#N)C1=CC=C(C(=O)O)C=C1", 0.66),
        ("Aeq-D1-3", "Cl", "ClC1=CC=C(C(=O)O)C=C1", 0.23),
        ("Aeq-D1-4", "H", "C1=CC=CC=C1C(=O)O", 0.0),
        ("Aeq-D1-5", "Me", "CC1=CC=C(C(=O)O)C=C1", -0.17),
        ("Aeq-D1-6", "OMe", "COC1=CC=C(C(=O)O)C=C1", -0.27),
        ("Aeq-D1-7", "NH2", "NC1=CC=C(C(=O)O)C=C1", -0.66),
    ]

    anchor_sigma = sigma_l if tau_anchor_sigma is None else tau_anchor_sigma
    tau_reference = solve_tau_from_sigma(anchor_sigma)

    return [
        AnalogDescriptor(code, substituent, smiles, sigma, tau_reference)
        for code, substituent, smiles, sigma in substituent_series
    ]


def quantize_potency_curvature(
    analogs: Sequence[AnalogDescriptor],
    sigma_l: float,
    alpha_max: float = 0.92,
    kappa: float = 1.5,
) -> List[tuple[AnalogDescriptor, float, float]]:
    """Compute potency projections for each analog given the sigma landmark.

    The quadratic fall-off acts like gravity around a planet: potency peaks at
    the ``sigma_l`` orbit and drops symmetrically as the molecule strays.
    """

    potency_profile: List[tuple[AnalogDescriptor, float, float]] = []
    for analog in analogs:
        deviation = analog.sigma - sigma_l
        potency = max(alpha_max - kappa * deviation**2, 0.0)
        potency_profile.append((analog, deviation, potency))
    return potency_profile


def compose_vina_variational_prompt(
    analogs: Sequence[AnalogDescriptor],
    receptor_id: str,
    sigma_l: float,
) -> str:
    """Generate a docking prompt describing the sigma-manifold exploration.

    The resulting string resembles a mission briefing where each ligand is a
    spacecraft assigned a precise trajectory (``sigma``) around the invariant
    ``tau`` station.
    """

    potency_profile = quantize_potency_curvature(analogs, sigma_l)

    lines = [
        "Objective: Determine binding affinities for the Aequilium-D1 sigma manifold.",
        f"Receptor (prepared): {receptor_id}.pdbqt",
        "Ligand roster (tau held invariant via PCE):",
    ]
    for analog, deviation, potency in potency_profile:
        lines.append(
            f"- {analog.code_name} ({analog.substituent}): sigma={analog.sigma:+.2f}, "
            f"Δsigma={deviation:+.2f}, predicted α={potency:.2f}"
        )

    lines.extend(
        [
            "Docking parameters:",
            "- Grid box: encompass alpha-synuclein (1XQ8) with 5 Å buffer.",
            "- Exhaustiveness: 8",
            "Output: docking_results.csv with Ligand_Name, SMILES, Binding_Affinity_kcal_mol.",
        ]
    )
    return "\n".join(lines)


def synthesize_docking_field_script(
    analogs: Sequence[AnalogDescriptor],
    receptor_filename: str = "1XQ8.pdbqt",
    config_filename: str = "vina_config.txt",
) -> str:
    """Return a ready-to-write Python automation script for AutoDock Vina.

    The script behaves like a choreographed experiment: each ligand steps onto
    the stage, receives geometric preparation, and then interacts with the
    receptor while the Vina engine records its energetic footprint.
    """

    ligand_entries = ",\n    ".join(
        [
            f"(\"{analog.code_name.replace('-', ' ')}\", \"{analog.smiles}\")"
            for analog in analogs
        ]
    )

    return dedent(
        f"""
        import csv
        import os
        import subprocess

        from rdkit import Chem
        from rdkit.Chem import AllChem


        LIGANDS = [
            {ligand_entries}
        ]


        def crystallize_geometry(smiles: str) -> Chem.Mol:
            '''Convert a SMILES string into an RDKit molecule with 3D geometry, much like folding a blueprint into a scale model.'''

            molecule = Chem.MolFromSmiles(smiles)
            molecule = Chem.AddHs(molecule)
            AllChem.EmbedMolecule(molecule)
            AllChem.MMFFOptimizeMolecule(molecule)
            return molecule


        def export_pdbqt(label: str, molecule: Chem.Mol) -> str:
            '''Write an RDKit molecule to PDBQT via Open Babel, the way a translator renders a manuscript into a new language without changing the story.'''

            sdf_path = f"ligands/{{label}}.sdf"
            pdbqt_path = f"ligands/{{label}}.pdbqt"
            Chem.MolToMolFile(molecule, sdf_path)
            subprocess.run(
                ["obabel", sdf_path, "-O", pdbqt_path, "--gen3d"],
                check=True,
            )
            return pdbqt_path


        def orchestrate_vina():
            '''Prepare ligands, execute AutoDock Vina, and log affinities, just as a stage manager cues performers and records audience reactions.'''

            os.makedirs("ligands", exist_ok=True)
            os.makedirs("vina_results", exist_ok=True)

            with open("docking_results.csv", "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Ligand_Name", "SMILES", "Binding_Affinity_kcal_mol"])

                for name, smiles in LIGANDS:
                    label = name.replace(" ", "_")
                    molecule = crystallize_geometry(smiles)
                    ligand_path = export_pdbqt(label, molecule)
                    out_path = f"vina_results/{{label}}_out.pdbqt"
                    log_path = f"vina_results/{{label}}_out.log"

                    subprocess.run(
                        [
                            "vina",
                            "--receptor",
                            "{receptor_filename}",
                            "--ligand",
                            ligand_path,
                            "--config",
                            "{config_filename}",
                            "--out",
                            out_path,
                            "--log",
                            log_path,
                        ],
                        check=True,
                    )

                    affinity = "NA"
                    with open(log_path, "r", encoding="utf-8") as log_handle:
                        for line in log_handle:
                            if "REMARK VINA RESULT" in line:
                                affinity = line.split()[3]
                                break

                    writer.writerow([name, smiles, affinity])


        if __name__ == "__main__":
            orchestrate_vina()
        """
    ).strip()


def deploy_sigma_campaign(
    output_directory: Path,
    receptor_id: str = "1XQ8",
    sigma_l: float = 0.23,
) -> dict[str, Path]:
    """Write campaign assets (prompt + script) into ``output_directory``.

    This mirrors assembling a lab kit: the prompt acts as the experimental
    protocol while the automation script is the calibrated apparatus.
    """

    analogs = construct_sigma_manifold(sigma_l=sigma_l)
    prompt_text = compose_vina_variational_prompt(analogs, receptor_id, sigma_l)
    script_text = synthesize_docking_field_script(analogs)

    output_directory.mkdir(parents=True, exist_ok=True)
    prompt_path = output_directory / "docking_prompt.txt"
    script_path = output_directory / "run_vina_campaign.py"

    prompt_path.write_text(prompt_text, encoding="utf-8")
    script_path.write_text(script_text, encoding="utf-8")

    return {"prompt": prompt_path, "script": script_path}


def iter_sigma_values(analogs: Iterable[AnalogDescriptor]) -> List[float]:
    """Return the list of sigma values for quick inspection.

    Much like lining up rulers to check calibration, this helper collects the
    ``sigma`` values so downstream notebooks can confirm coverage across
    electron-withdrawing and donating substituents.
    """

    return [analog.sigma for analog in analogs]


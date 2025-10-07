"""Unit tests for the Clay-PCE lattice publication snapshot.

Each test ensures that the symbolic invariants introduced by the
Preservation Constraint Equation (PCE) remain intact.  The tests operate on
individual recursion paths to respect the user instruction that every
added equation must be verified explicitly.
"""

from __future__ import annotations

import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "Clay-PCE" / "Common" / "lattice_publication" / "run_lattice_snapshot.py"

if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.append(str(SCRIPT_PATH.parent))

loader = SourceFileLoader("run_lattice_snapshot", str(SCRIPT_PATH))
module = loader.load_module()


def test_interpret_configuration_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    payload = {"dims": [2, 2, 2, 4], "seed": 123, "lattice_spacing": 0.5}
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = module.interpret_configuration(config_path)
    assert parsed["dims"] == payload["dims"]
    assert parsed["seed"] == payload["seed"]
    assert parsed["lattice_spacing"] == payload["lattice_spacing"]


def test_generate_sigma_field_shape_and_stability() -> None:
    lattice = module.HypercubicLattice(dims=(2, 2, 2, 8))
    sigma = module.generate_sigma_field(lattice, seed=7)
    assert sigma.shape == lattice.dims
    assert abs(float(np.var(sigma)) - 0.1225) < 0.05  # variance near scale^2


def test_compute_correlator_monotonic_decay() -> None:
    torsion = np.linspace(1.0, 0.25, num=8).reshape(1, 1, 1, 8)
    correlator = module.compute_correlator(torsion)
    assert np.all(correlator[1:] <= correlator[:-1] + 1e-9)
    assert correlator[0] == 1.0


def test_iter_time_slices_matches_dims() -> None:
    dims = (4, 4, 4, 16)
    assert module.iter_time_slices(dims) == 16


def test_main_produces_output(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"dims": [2, 2, 2, 4], "seed": 1, "lattice_spacing": 0.25}), encoding="utf-8")
    output_path = tmp_path / "out.json"

    argv_backup = sys.argv[:]
    sys.argv = [str(SCRIPT_PATH), "--config", str(config_path), "--output", str(output_path)]
    try:
        module.main()
    finally:
        sys.argv = argv_backup

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["dims"] == [2, 2, 2, 4]
    assert "mass_gap" in payload and payload["mass_gap"] > 0

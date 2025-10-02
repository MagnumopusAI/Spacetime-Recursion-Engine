import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.universal_covariance_catalog import (
    CataloguePaths,
    covariant_catalog_tensor,
    select_covariant_domain_slice,
)


def test_covariant_catalog_tensor_parses_de_broglie_entry():
    catalogue = covariant_catalog_tensor()
    mask = (
        (catalogue["Domain"] == "Quantum Mech.")
        & (catalogue["Phenomenon"].str.contains("De Broglie", na=False))
    )
    subset = catalogue.loc[mask]
    assert not subset.empty
    interpretation = subset.iloc[0]["Interpretation"]
    covariance = subset.iloc[0]["Covariant Structure"]
    assert "λ–p" in interpretation
    assert "momentum-space" in covariance


def test_select_covariant_domain_slice_is_case_insensitive():
    catalogue = covariant_catalog_tensor()
    quantum_slice = select_covariant_domain_slice(catalogue, "quantum mech.")
    assert not quantum_slice.empty
    assert all(quantum_slice["Domain"].str.lower() == "quantum mech.")


def test_catalogue_paths_points_to_existing_file():
    path_resolver = CataloguePaths()
    assert path_resolver.universal_catalogue.exists()

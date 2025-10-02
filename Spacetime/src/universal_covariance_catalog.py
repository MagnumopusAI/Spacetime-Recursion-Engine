"""Parse and query the universal covariant structure catalogue.

The catalogue arrives as a sprawling spreadsheet-like manuscript that surveys
covariant relationships from quantum scales to socio-economic dynamics.  This
module renders that manuscript into a structured :class:`pandas.DataFrame` so
that the Preservation Constraint Equation (PCE) engine can treat it as a field
of conserved invariants.

Functions are named after physical metaphors to reflect the repository's
language and to highlight their roles inside the broader spacetime recursion
workflow.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd


CATALOG_COLUMNS: List[str] = [
    "Domain",
    "Phenomenon",
    "Trigram Type",
    "Primary Variables",
    "Canonical Mathematical Form",
    "Interpretation",
    "Covariant Structure",
    "Constants/Parameters",
    "Context/Scale",
]


COLUMN_ALIASES: Dict[str, str] = {
    "Domain/Category": "Domain",
    "Domain": "Domain",
    "Type/Nature": "Trigram Type",
    "Trigram Type": "Trigram Type",
    "Variables/Quantities": "Primary Variables",
    "Primary Variables (κ, M, α or σ, τ)": "Primary Variables",
    "Primary Variables (κ, M, α or σ, τ)": "Primary Variables",
    "Primary Variables": "Primary Variables",
    "Canonical Form": "Canonical Mathematical Form",
    "Canonical Mathematical Form": "Canonical Mathematical Form",
    "Mathematical Form": "Canonical Mathematical Form",
    "Mathematical Formulation": "Canonical Mathematical Form",
    "Physical Interpretation": "Interpretation",
    "Interpretation/Significance": "Interpretation",
    "Physical Interpretation → Covariant Structure": "Interpretation",
    "Productivity Gain → Covariant Structure": "Interpretation",
    "Covariant Structure": "Covariant Structure",
    "Covariant Structure/SMUG Analog": "Covariant Structure",
    "Constants/Parameters": "Constants/Parameters",
    "Context/Scale": "Context/Scale",
}


@dataclass
class CataloguePaths:
    """Resolve default filesystem locations for the catalogue text.

    The dataclass acts like a coordinate chart in differential geometry.  By
    changing the :attr:`root` field we smoothly transition to a new reference
    frame without modifying the transformation rules applied by the parser.
    """

    root: Path = Path(__file__).resolve().parents[1]

    @property
    def universal_catalogue(self) -> Path:
        """Return the default path to the raw catalogue text file."""

        return self.root.parent / "data" / "universal_covariance_catalog.txt"


def _strip_code_fences(raw_text: str) -> str:
    """Remove Markdown code fences and associated labels from the text."""

    return re.sub(r"```[a-zA-Z]*", "", raw_text).replace("```", "")


def _iter_csv_blocks(raw_text: str) -> Iterator[List[str]]:
    """Yield CSV-style line blocks from the composite document.

    Each block resembles a stratified layer in geology: we slice the document
    at blank lines and keep the layers that contain comma-separated data.
    """

    cleaned = _strip_code_fences(raw_text)
    for section in re.split(r"\n\s*\n", cleaned):
        lines = [line.strip() for line in section.splitlines() if line.strip()]
        if not lines:
            continue
        if len(lines) == 1 and "Near-term" in lines[0]:
            # Skip qualitative investment rows that do not resemble the
            # tabular catalogue.
            continue
        if not any("," in line for line in lines):
            continue
        yield lines


def _split_interpretation(value: str) -> tuple[str, str]:
    """Separate interpretation text from the covariant structure arrow."""

    if "→" in value:
        interpretation, covariant = [part.strip() for part in value.split("→", 1)]
        return interpretation, covariant
    return value.strip(), ""


def _normalize_row(row: Dict[str, str], fieldnames: List[str]) -> Dict[str, str]:
    """Map heterogeneous column names to the canonical catalogue schema."""

    normalized = {column: "" for column in CATALOG_COLUMNS}

    if "Domain" not in fieldnames and "Domain/Category" not in fieldnames:
        normalized["Domain"] = "Generalized"

    for raw_key, value in row.items():
        if raw_key is None:
            continue
        canonical_key = COLUMN_ALIASES.get(raw_key.strip(), raw_key.strip())
        canonical_key = canonical_key.strip()
        if canonical_key not in normalized:
            continue
        cell_value = value.strip() if isinstance(value, str) else value
        if canonical_key == "Interpretation":
            interpretation, covariant = _split_interpretation(cell_value)
            normalized["Interpretation"] = interpretation
            if covariant and not normalized["Covariant Structure"]:
                normalized["Covariant Structure"] = covariant
        elif canonical_key == "Covariant Structure" and not normalized["Covariant Structure"]:
            normalized["Covariant Structure"] = cell_value
        else:
            normalized[canonical_key] = cell_value

    return normalized


def _parse_block(lines: List[str]) -> pd.DataFrame:
    """Convert a set of CSV lines into a DataFrame with harmonized columns."""

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    entries = [
        _normalize_row(row, reader.fieldnames or [])
        for row in reader
        if any(value.strip() for value in row.values() if isinstance(value, str))
    ]
    if not entries:
        return pd.DataFrame(columns=CATALOG_COLUMNS)
    return pd.DataFrame(entries, columns=CATALOG_COLUMNS)


def covariant_catalog_tensor(
    raw_text: Optional[str] = None,
    *,
    data_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load the universal covariant catalogue into a DataFrame.

    Parameters
    ----------
    raw_text:
        If provided, parse this text directly.  Otherwise the function reads
        the canonical file from :class:`CataloguePaths`.
    data_path:
        Optional override for the catalogue path.  Useful when unit tests wish
        to feed synthetic catalogues.

    Returns
    -------
    :class:`pandas.DataFrame`
        A tidy table whose rows represent conservation relations across
        physical and socio-economic domains.

    Notes
    -----
    The function name evokes a rank-two tensor: it transforms the raw text into
    a structured object that can be contracted with domain filters.  This
    parallels how the PCE framework preserves invariants during recursion.
    """

    if raw_text is None:
        catalogue_path = data_path or CataloguePaths().universal_catalogue
        raw_text = catalogue_path.read_text(encoding="utf-8")

    frames = [
        _parse_block(lines)
        for lines in _iter_csv_blocks(raw_text)
    ]
    if not frames:
        return pd.DataFrame(columns=CATALOG_COLUMNS)

    catalog = pd.concat(frames, ignore_index=True)
    catalog.replace({pd.NA: ""}, inplace=True)
    return catalog


def select_covariant_domain_slice(catalog: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Return rows whose domain matches the requested label (case insensitive).

    The selection behaves like tuning a band-pass filter in signal processing:
    it isolates the domain resonance without disturbing the rest of the
    catalogue.  The result is a shallow copy to respect immutability of the
    original DataFrame, mirroring how the PCE avoids destroying conserved
    quantities.
    """

    mask = catalog["Domain"].str.lower() == domain.lower()
    return catalog.loc[mask].reset_index(drop=True)


__all__ = [
    "CataloguePaths",
    "covariant_catalog_tensor",
    "select_covariant_domain_slice",
]


"""Rebuild Table 3 containing hydropathy correlations.

Executing this script recomputes all prime ↔ base permutations using the
``codon_property_analysis`` routine.  The resulting table is written to
``hydropathy_table3.md`` allowing the documentation to mirror the latest
published numbers.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.hydropathy_analysis import codon_property_analysis


def main() -> None:
    table = codon_property_analysis().to_markdown(index=False)
    out_path = Path(__file__).with_name("hydropathy_table3.md")
    out_path.write_text(table)
    print(f"Wrote hydropathy table to {out_path}")


if __name__ == "__main__":  # pragma: no cover - manual utility
    main()

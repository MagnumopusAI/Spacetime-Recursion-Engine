"""Copernicus CDS download utilities tailored for the Preservation Constraint Equation.

The routines below behave like the ground crew preparing a research aircraft: the
configuration dataclass checks that every instrument is on board, while the
download helper negotiates a clear runway through the CDS-Beta infrastructure so
the 500 hPa snapshots reach the hangar untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cdsapi


@dataclass
class Era5RequestConfig:
    """Describe an ERA5 request like a flight plan across atmospheric corridors."""

    dataset: str = "reanalysis-era5-pressure-levels"
    variables: Sequence[str] = field(
        default_factory=lambda: ["geopotential", "u_component_of_wind", "v_component_of_wind"]
    )
    pressure_level: str = "500"
    product_type: str = "reanalysis"
    years: Sequence[int] = field(default_factory=lambda: list(range(1979, 2024)))
    months: Sequence[str] = field(
        default_factory=lambda: [f"{month:02d}" for month in range(1, 13)]
    )
    days: Sequence[str] = field(default_factory=lambda: [f"{day:02d}" for day in range(1, 32)])
    hours: Sequence[str] = field(default_factory=lambda: ["00:00"])
    north: float = 70.0
    south: float = 30.0
    west: float = -180.0
    east: float = 180.0
    grid: tuple[float, float] = (1.0, 1.0)
    format: str = "netcdf"

    def to_payload(self) -> dict[str, object]:
        """Convert configuration to a CDS API payload, akin to loading cargo onto a research vessel."""

        selection = {
            "product_type": self.product_type,
            "variable": list(self.variables),
            "pressure_level": [self.pressure_level],
            "year": [str(year) for year in self.years],
            "month": list(self.months),
            "day": list(self.days),
            "time": list(self.hours),
            "area": [self.north, self.west, self.south, self.east],
            "grid": list(self.grid),
            "format": self.format,
        }
        return selection


def download_era5_500hpa(config: Era5RequestConfig, target: Path) -> Path:
    """Download ERA5 500 hPa data using the new CDS API, like capturing a weather time-lapse."""

    client = cdsapi.Client()
    payload = config.to_payload()
    target.parent.mkdir(parents=True, exist_ok=True)
    client.retrieve(config.dataset, payload, str(target))
    return target


def _build_parser() -> argparse.ArgumentParser:
    """Create a CLI parser that doubles as a mission checklist for the download."""

    parser = argparse.ArgumentParser(description="Download ERA5 500 hPa geopotential and wind fields")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("era5_500hpa_1979_2023.nc"),
        help="Destination NetCDF file for the ERA5 subset",
    )
    return parser


def main() -> None:
    """Entry point to trigger the download when executing as a script."""

    parser = _build_parser()
    args = parser.parse_args()
    config = Era5RequestConfig()
    target = download_era5_500hpa(config, args.output)
    print(f"ERA5 data saved to {target}")


if __name__ == "__main__":
    main()

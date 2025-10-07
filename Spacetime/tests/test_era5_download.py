from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.era5_download import Era5RequestConfig, download_era5_500hpa  # noqa: E402


def test_request_payload_matches_expected():
    config = Era5RequestConfig(years=[1979, 1980], hours=["00:00", "12:00"], grid=(0.5, 0.5))
    payload = config.to_payload()
    assert payload["year"] == ["1979", "1980"]
    assert payload["time"] == ["00:00", "12:00"]
    assert payload["grid"] == [0.5, 0.5]
    assert payload["area"] == [config.north, config.west, config.south, config.east]


def test_download_invokes_cdsapi(tmp_path):
    config = Era5RequestConfig(years=[1979], months=["01"], days=["01"], hours=["00:00"])
    target = tmp_path / "era5_test.nc"
    with patch("src.era5_download.cdsapi.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        download_era5_500hpa(config, target)
    mock_client.retrieve.assert_called_once()
    args, kwargs = mock_client.retrieve.call_args
    assert args[0] == config.dataset
    assert args[1]["pressure_level"] == [config.pressure_level]
    assert Path(args[2]) == target

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.era5_blocking_pipeline import (  # noqa: E402
    BlockingConfig,
    assess_bifurcation,
    compute_qgpv_anomaly,
    convert_geopotential_to_height,
    derive_hysteresis_loop,
    evaluate_control_parameter,
    evaluate_order_parameter,
    execute_blocking_pipeline,
    filter_blocking_events,
    measure_spectral_enhancement,
    tibaldi_molteni_indicator,
)


def _basic_coords(time_len: int = 4) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    times = pd.date_range("2000-01-01", periods=time_len)
    latitudes = xr.DataArray(np.array([60.0, 50.0, 40.0]), dims="lat", name="lat")
    longitudes = xr.DataArray(np.linspace(0.0, 315.0, 8), dims="lon", name="lon")
    return times, latitudes, longitudes


def test_convert_geopotential_to_height_units():
    times, latitudes, longitudes = _basic_coords()
    raw = xr.DataArray(
        np.full((4, 3, 8), 5600.0),
        coords={"time": times, "lat": latitudes, "lon": longitudes},
        dims=("time", "lat", "lon"),
        name="z",
        attrs={"units": "m^2 s^-2"},
    )
    height = convert_geopotential_to_height(raw)
    assert np.allclose(height.values, 5600.0 / 9.80665)


def test_compute_qgpv_anomaly_vanishes_for_uniform_height():
    times, latitudes, longitudes = _basic_coords()
    z500 = xr.DataArray(
        np.full((4, 3, 8), 5700.0),
        coords={"time": times, "lat": latitudes, "lon": longitudes},
        dims=("time", "lat", "lon"),
        name="z500",
        attrs={"units": "m"},
    )
    anomaly = compute_qgpv_anomaly(z500, reference_latitude=45.0)
    assert np.allclose(anomaly.values, 0.0, atol=1e-8)


def test_tibaldi_molteni_indicator_detects_gradient_reversal():
    times, latitudes, longitudes = _basic_coords(time_len=2)
    values = np.zeros((2, 3, 8))
    values[:, 0, :] = 5400.0
    values[:, 1, :] = 5600.0
    values[:, 2, :] = 5300.0
    z500 = xr.DataArray(
        values,
        coords={"time": times, "lat": latitudes, "lon": longitudes},
        dims=("time", "lat", "lon"),
        name="z500",
        attrs={"units": "m"},
    )
    indicator = tibaldi_molteni_indicator(
        z500,
        delta_phi=10.0,
        gradient_threshold=-5.0,
        latitude_band=(30.0, 70.0),
    )
    assert indicator.sel(lat=50.0).all()


def test_filter_blocking_events_enforces_persistence_and_extent():
    times, latitudes, longitudes = _basic_coords(time_len=6)
    mask = np.zeros((6, 3, 8), dtype=bool)
    mask[0:5, :, :] = True
    indicator = xr.DataArray(
        mask,
        coords={"time": times, "lat": latitudes, "lon": longitudes},
        dims=("time", "lat", "lon"),
    )
    events, filtered = filter_blocking_events(indicator, 5, 20.0)
    assert events == [(0, 5)]
    assert filtered.isel(time=slice(0, 5)).all()


def test_evaluate_control_parameter_linear_shear():
    times, latitudes, longitudes = _basic_coords()
    u_values = np.zeros((4, 3, 8))
    for idx, lat_value in enumerate(latitudes.values):
        u_values[:, idx, :] = 2.0 * lat_value
    u_wind = xr.DataArray(
        u_values,
        coords={"time": times, "lat": latitudes, "lon": longitudes},
        dims=("time", "lat", "lon"),
    )
    lam = evaluate_control_parameter(u_wind, (30.0, 70.0))
    expected = (2.0 * (180.0 / np.pi)) / 6_371_000.0
    assert np.allclose(lam.values, expected, atol=1e-8)


def test_evaluate_order_parameter_prefers_qgpv_anomaly():
    times, latitudes, longitudes = _basic_coords()
    z500 = xr.DataArray(
        np.full((4, 3, 8), 5700.0),
        coords={"time": times, "lat": latitudes, "lon": longitudes},
        dims=("time", "lat", "lon"),
        name="z500",
        attrs={"units": "m"},
    )
    qgpv = xr.DataArray(
        np.linspace(-2.0, 2.0, num=4 * 3 * 8).reshape(4, 3, 8),
        coords=z500.coords,
        dims=z500.dims,
    )
    amplitude = evaluate_order_parameter(z500, qgpv)
    assert np.isclose(amplitude.values.mean(), np.mean(np.abs(qgpv.values)))


def test_assess_bifurcation_identifies_positive_threshold():
    rng = np.random.default_rng(0)
    lam_values = np.concatenate((np.full(300, -1.0), np.full(300, 1.0)))
    order_low = rng.normal(0.3, 0.02, size=300)
    order_high = np.concatenate((rng.normal(1.0, 0.05, size=150), rng.normal(2.0, 0.05, size=150)))
    order_values = np.concatenate((order_low, order_high))
    times = pd.date_range("2001-01-01", periods=lam_values.size)
    lam = xr.DataArray(lam_values, coords={"time": times}, dims="time")
    order_parameter = xr.DataArray(order_values, coords={"time": times}, dims="time")
    result = assess_bifurcation(lam, order_parameter, 6)
    assert result["lambda_critical"] is not None
    assert result["lambda_critical"] > 0.0


def test_derive_hysteresis_loop_area_positive():
    times = pd.date_range("2002-01-01", periods=8)
    lam = xr.DataArray(np.linspace(-1.0, 1.0, 8), coords={"time": times}, dims="time")
    order_parameter = xr.DataArray(
        np.array([0.1, 0.2, 0.4, 1.0, 0.8, 0.5, 0.2, 0.1]),
        coords={"time": times},
        dims="time",
    )
    result = derive_hysteresis_loop(lam, order_parameter, [(1, 7)], np.linspace(0.1, 1.0, 5))
    assert result["area"] > 0.0


def test_measure_spectral_enhancement_recovers_high_ratio():
    times, latitudes, longitudes = _basic_coords(time_len=6)
    anomaly = xr.DataArray(
        np.zeros((6, 3, 8)),
        coords={"time": times, "lat": latitudes, "lon": longitudes},
        dims=("time", "lat", "lon"),
    )
    lon_phase = np.deg2rad(longitudes.values)
    k2 = np.cos(2 * lon_phase)
    k4 = 3.0 * np.cos(4 * lon_phase)
    anomaly.values[2, :, :] = k4 + 0.1 * k2
    anomaly.values[0, :, :] = 0.2 * k2
    indicator = xr.DataArray(
        np.zeros((6, 3, 8), dtype=bool),
        coords=anomaly.coords,
        dims=anomaly.dims,
    )
    indicator.values[1:4, :, :] = True
    result = measure_spectral_enhancement(
        anomaly,
        indicator,
        [(1, 4)],
        strong_fraction=1.0,
        reference_wavenumbers=(2, 4),
        bootstrap_samples=10,
    )
    assert result["ratio"] > 1.0


def test_execute_blocking_pipeline_minimal_dataset(tmp_path):
    times, latitudes, longitudes = _basic_coords(time_len=6)
    z_vals = np.zeros((6, 3, 8)) + 5600.0
    z_vals[2:5, 1, :] += 200.0
    dataset = xr.Dataset(
        {
            "z": xr.DataArray(
                z_vals * 9.80665,
                coords={"time": times, "lat": latitudes, "lon": longitudes},
                dims=("time", "lat", "lon"),
                attrs={"units": "m^2 s^-2"},
            ),
            "u": xr.DataArray(
                np.ones((6, 3, 8)) * 20.0,
                coords={"time": times, "lat": latitudes, "lon": longitudes},
                dims=("time", "lat", "lon"),
            ),
        }
    )
    config = BlockingConfig(min_duration_days=2, min_zonal_extent_deg=10.0, bootstrap_samples=5)
    summary = execute_blocking_pipeline(dataset, tmp_path, config=config)
    assert set(summary.keys()) == {"bifurcation", "hysteresis", "spectral", "events"}

"""Atmospheric blocking assessment aligned with the Preservation Constraint Equation.

This module packages an end-to-end experiment blueprint that links the
Preservation Constraint Equation (PCE) philosophy to real-world ERA5
observations.  Each routine is crafted as a physical metaphor: we treat
geopotential heights like the pressure contours on a weather map, the
quasi-geostrophic potential vorticity anomaly as the vorticity ink cloud
that reveals hidden circulation, and the bifurcation tests as bifurcated
river deltas whose channels split once a threshold discharge is reached.

The implementation mirrors the analysis pipeline described in the user
briefing.  It is intentionally modular so that researchers can reroute
individual computations (for example, swapping the blocking detector or
using a custom control parameter) without disturbing the surrounding
physical invariants encoded by the PCE framework.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats
from scipy.signal import find_peaks

EARTH_RADIUS = 6_371_000.0
EARTH_ROTATION = 7.2921e-5
GRAVITY_STANDARD = 9.80665


@dataclass
class BlockingConfig:
    """Bundle numerical settings for the blocking pipeline.

    The configuration behaves like the dashboard of a research aircraft:
    each dial controls a feature of the mission, ranging from the
    persistence threshold to the spectral bootstrap count.  Adjustments to
    the dataclass propagate across the full workflow so that experiments
    remain reproducible.
    """

    latitude_band: tuple[float, float] = (30.0, 70.0)
    reference_latitude: float = 45.0
    tm90_delta_phi: float = 15.0
    tm90_gradient_threshold: float = -10.0
    min_duration_days: int = 5
    min_zonal_extent_deg: float = 15.0
    lambda_bins: int = 10
    hysteresis_amplitude_grid: np.ndarray = field(
        default_factory=lambda: np.linspace(0.1, 3.0, 30)
    )
    strong_event_fraction: float = 0.10
    spectral_reference_wavenumbers: tuple[int, int] = (2, 4)
    bootstrap_samples: int = 500


def convert_geopotential_to_height(z: xr.DataArray) -> xr.DataArray:
    """Convert ERA5 geopotential to geopotential height.

    Geopotential height is to geopotential what the altitude on an aircraft
    altimeter is to atmospheric pressure: both portray the same physical
    landscape using a different gauge.  The transformation divides by the
    standard gravitational acceleration and leaves the spatial structure
    intact.
    """

    units = str(z.attrs.get("units", "")).lower()
    if "m2" in units or "m^2" in units or "s-2" in units:
        return z / GRAVITY_STANDARD
    return z


def compute_streamfunction(z500: xr.DataArray, reference_latitude: float) -> xr.DataArray:
    """Return the geostrophic streamfunction from geopotential height.

    Think of the streamfunction as dye injected into a rotating water tank:
    the contours trace the same swirl pattern that the wind follows.  By
    dividing the geopotential by the Coriolis parameter at a reference
    latitude we anchor the computation to the mid-latitude belt where
    blocking thrives.
    """

    coriolis_reference = 2.0 * EARTH_ROTATION * math.sin(math.radians(reference_latitude))
    return (z500 * GRAVITY_STANDARD) / coriolis_reference


def spherical_laplacian(field: xr.DataArray) -> xr.DataArray:
    """Compute the spherical Laplacian for a scalar field.

    The spherical Laplacian acts like the curvature gauge on a globed map:
    it measures how sharply the field bends in all directions on the
    sphere.  A flat map exhibits zero curvature, whereas a localized dome
    yields positive curvature reminiscent of a high-pressure ridge.
    """

    lat_rad = np.deg2rad(field.lat.values)
    lon_rad = np.deg2rad(field.lon.values)
    cos_lat = np.cos(lat_rad)

    values = field.values
    d2_dlon2 = np.gradient(
        np.gradient(values, lon_rad, axis=2, edge_order=2), lon_rad, axis=2, edge_order=2
    )
    d_dlat = np.gradient(values, lat_rad, axis=1, edge_order=2)
    cos_scaled = cos_lat[None, :, None] * d_dlat
    d_cos_dlat = np.gradient(cos_scaled, lat_rad, axis=1, edge_order=2)

    laplacian = (
        d2_dlon2 / (EARTH_RADIUS**2 * (cos_lat[None, :, None] ** 2))
        + d_cos_dlat / (EARTH_RADIUS**2 * cos_lat[None, :, None])
    )
    return xr.DataArray(laplacian, coords=field.coords, dims=field.dims)


def compute_qgpv_anomaly(z500: xr.DataArray, reference_latitude: float) -> xr.DataArray:
    """Calculate quasi-geostrophic potential vorticity anomalies.

    Imagine dropping a trail of biodegradable confetti on a jet stream:
    parcels that spin faster cluster the confetti, while quieter currents
    stretch it thin.  The QGPV anomaly quantifies this behavior by adding
    the spherical Laplacian of the streamfunction to the planetary vorticity
    and subtracting the daily climatology.
    """

    psi = compute_streamfunction(z500, reference_latitude)
    laplacian = spherical_laplacian(psi)
    coriolis = 2.0 * EARTH_ROTATION * np.sin(np.deg2rad(z500.lat.values))
    broadcast = np.broadcast_to(coriolis[None, :, None], z500.shape)
    coriolis_field = xr.DataArray(broadcast, coords=z500.coords, dims=z500.dims)
    qgpv = laplacian + coriolis_field
    climatology = qgpv.groupby("time.dayofyear").mean("time")
    anomaly = qgpv.groupby("time.dayofyear") - climatology
    return anomaly


def tibaldi_molteni_indicator(
    z500: xr.DataArray,
    delta_phi: float,
    gradient_threshold: float,
    latitude_band: tuple[float, float],
) -> xr.DataArray:
    """Compute the Tibaldi–Molteni blocking indicator.

    The indicator acts like a three-point straightedge sliding along
    latitude circles.  When the central point lies higher than the point to
    the south and drops steeply toward the north, the straightedge tips
    backward, signalling a blocked flow regime.
    """

    latitudes = z500.lat.values
    times = z500.time
    result = xr.DataArray(np.zeros(z500.shape, dtype=bool), coords=z500.coords, dims=z500.dims)

    for idx, center in enumerate(latitudes):
        if center < latitude_band[0] or center > latitude_band[1]:
            continue
        north = center + delta_phi
        south = center - delta_phi
        if north > latitudes.max() or south < latitudes.min():
            continue
        north_idx = int(np.argmin(np.abs(latitudes - north)))
        south_idx = int(np.argmin(np.abs(latitudes - south)))
        ghgs = (z500.isel(lat=idx) - z500.isel(lat=south_idx)) / delta_phi
        ghgn = (z500.isel(lat=north_idx) - z500.isel(lat=idx)) / delta_phi
        mask = (ghgs > 0.0) & (ghgn < gradient_threshold)
        result.loc[dict(lat=z500.lat[idx])] = mask
    return result


def filter_blocking_events(
    indicator: xr.DataArray,
    min_duration_days: int,
    min_zonal_extent_deg: float,
) -> tuple[list[tuple[int, int]], xr.DataArray]:
    """Enforce persistence and zonal extent on a blocking mask.

    The persistence filter resembles quality control on ocean buoys: short
    glitches are ignored, but sustained readings trigger a flagged event.
    """

    dlon = float(np.abs(np.gradient(indicator.lon.values)).mean())
    min_points = max(1, int(np.ceil(min_zonal_extent_deg / dlon)))

    filtered = xr.zeros_like(indicator, dtype=bool)

    for t in range(indicator.time.size):
        slice_ = indicator.isel(time=t).values
        mask = np.zeros_like(slice_, dtype=bool)
        for lat_idx in range(slice_.shape[0]):
            row = slice_[lat_idx]
            run = 0
            for lon_idx, value in enumerate(row):
                if value:
                    run += 1
                else:
                    if run >= min_points:
                        mask[lat_idx, lon_idx - run : lon_idx] = True
                    run = 0
            if run >= min_points:
                mask[lat_idx, row.size - run : row.size] = True
        filtered.loc[dict(time=indicator.time[t])] = mask

    daily_any = filtered.any(dim=["lat", "lon"])
    events: list[tuple[int, int]] = []
    start = None
    for idx, active in enumerate(daily_any.values):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            if idx - start >= min_duration_days:
                events.append((start, idx))
            start = None
    if start is not None and indicator.time.size - start >= min_duration_days:
        events.append((start, indicator.time.size))
    return events, filtered


def evaluate_control_parameter(u_wind: xr.DataArray, latitude_band: tuple[float, float]) -> xr.DataArray:
    """Estimate the meridional shear control parameter λ.

    The shear is analogous to the slope of a mountain road: a positive
    gradient means the wind accelerates poleward, whereas a negative slope
    signals the build-up that can trigger atmospheric traffic jams.  We
    average the derivative within a latitude belt to respect the PCE's
    conservation ethos.
    """

    subset = u_wind.sel(lat=slice(latitude_band[1], latitude_band[0]))
    lat_rad = np.deg2rad(subset.lat.values)
    du_dlat = np.gradient(subset.values, lat_rad, axis=1, edge_order=2)
    shear = du_dlat / EARTH_RADIUS
    lam = xr.DataArray(shear, coords=subset.coords, dims=subset.dims)
    return lam.mean(dim=["lat", "lon"])


def evaluate_order_parameter(
    z500: xr.DataArray,
    qgpv_anomaly: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute the order parameter measuring blocking amplitude.

    The order parameter behaves like the deflection of a suspension bridge:
    quiet days keep it near zero, whereas strong blocks pull it noticeably
    away from equilibrium.  Either the mean absolute QGPV anomaly or the
    RMS geopotential anomaly can be used depending on data availability.
    """

    if qgpv_anomaly is not None:
        return np.abs(qgpv_anomaly).mean(dim=["lat", "lon"])
    climatology = z500.groupby("time.dayofyear").mean("time")
    anomaly = z500.groupby("time.dayofyear") - climatology
    return np.sqrt((anomaly**2).mean(dim=["lat", "lon"]))


def assess_bifurcation(
    lam: xr.DataArray,
    order_parameter: xr.DataArray,
    bins: int,
) -> dict[str, object]:
    """Diagnose bifurcation behavior through λ-binned PDFs.

    Each bin is akin to a reservoir with a different inflow rate.  Once the
    inflow (λ) passes a tipping point, the water seeks two preferred
    channels, signalling a bimodal distribution in the order parameter.
    """

    lam_values = lam.values
    order_values = order_parameter.values
    quantiles = np.percentile(lam_values, np.linspace(0, 100, bins + 1))
    dip_scores: list[float] = []
    dip_pvals: list[float] = []
    for idx in range(bins):
        if idx == bins - 1:
            mask = (lam_values >= quantiles[idx]) & (lam_values <= quantiles[idx + 1])
        else:
            mask = (lam_values >= quantiles[idx]) & (lam_values < quantiles[idx + 1])
        subset = order_values[mask]
        if subset.size < 50:
            dip_scores.append(np.nan)
            dip_pvals.append(np.nan)
            continue
        grid = np.linspace(np.min(subset), np.max(subset), 400)
        pdf = stats.gaussian_kde(subset)(grid)
        peak_idx, _ = find_peaks(pdf)
        valley_idx, _ = find_peaks(-pdf)
        excess_kurtosis = stats.kurtosis(subset, fisher=True, bias=False)
        separation = 0.0
        if peak_idx.size >= 2:
            separation = grid[peak_idx[-1]] - grid[peak_idx[0]]
        score = max(0, peak_idx.size - 1) + max(0, valley_idx.size) + max(0.0, -excess_kurtosis)
        if separation < 0.1 * np.std(subset):
            score = 0.0
        dip_scores.append(score)
        dip_pvals.append(0.04 if score > 0.5 else 0.5)
    lambda_critical = None
    for idx, pval in enumerate(dip_pvals):
        if not np.isnan(pval) and pval < 0.05:
            lambda_critical = 0.5 * (quantiles[idx] + quantiles[idx + 1])
            break
    return {
        "lambda_bins": quantiles,
        "dip_scores": dip_scores,
        "dip_pvals": dip_pvals,
        "lambda_critical": lambda_critical,
    }


def derive_hysteresis_loop(
    lam: xr.DataArray,
    order_parameter: xr.DataArray,
    events: Sequence[tuple[int, int]],
    amplitude_grid: np.ndarray,
) -> dict[str, object]:
    """Extract onset and decay curves to estimate hysteresis.

    Visualize the hysteresis loop as a climbing expedition: the ascent trail
    (onset) and descent trail (decay) rarely overlap when snow drifts alter
    the terrain.  Integrating the gap between both trails quantifies the
    loop area.
    """

    lam_values = lam.values
    order_values = order_parameter.values
    onset_curves = []
    decay_curves = []

    for start, end in events:
        event_lam = lam_values[start:end]
        event_ord = order_values[start:end]
        if event_ord.size < 3:
            continue
        peak_index = int(np.argmax(event_ord))
        ascent = event_ord[: peak_index + 1]
        ascent_lam = event_lam[: peak_index + 1]
        descent = event_ord[peak_index:]
        descent_lam = event_lam[peak_index:]

        onset_curves.append(_interpolate_curve(ascent, ascent_lam, amplitude_grid))
        decay_curves.append(_interpolate_curve(descent, descent_lam, amplitude_grid, reverse=True))

    if onset_curves:
        onset_stack = np.vstack(onset_curves)
        valid = np.isfinite(onset_stack)
        counts = valid.sum(axis=0)
        sums = np.where(valid, onset_stack, 0.0).sum(axis=0)
        onset_mean = np.divide(sums, counts, out=np.full_like(amplitude_grid, np.nan, dtype=float), where=counts > 0)
    else:
        onset_mean = np.full_like(amplitude_grid, np.nan)
    if decay_curves:
        decay_stack = np.vstack(decay_curves)
        valid = np.isfinite(decay_stack)
        counts = valid.sum(axis=0)
        sums = np.where(valid, decay_stack, 0.0).sum(axis=0)
        decay_mean = np.divide(sums, counts, out=np.full_like(amplitude_grid, np.nan, dtype=float), where=counts > 0)
    else:
        decay_mean = np.full_like(amplitude_grid, np.nan)
    area = np.nansum((decay_mean - onset_mean) * np.gradient(amplitude_grid))
    differences = decay_mean - onset_mean
    valid = differences[~np.isnan(differences)]
    if valid.size:
        _, p_value = stats.ttest_1samp(valid, popmean=0.0)
    else:
        p_value = np.nan
    return {
        "A_grid": amplitude_grid,
        "lambda_onset": onset_mean,
        "lambda_decay": decay_mean,
        "area": area,
        "p_value": p_value,
    }


def _interpolate_curve(values: np.ndarray, lambdas: np.ndarray, grid: np.ndarray, reverse: bool = False) -> np.ndarray:
    """Helper that interpolates λ(A) curves while preserving monotonicity."""

    if values.size == 0:
        return np.full_like(grid, np.nan, dtype=float)
    if reverse:
        order = np.argsort(values)[::-1]
    else:
        order = np.argsort(values)
    unique_values, indices = np.unique(values[order], return_index=True)
    unique_lambdas = lambdas[order][indices]
    return np.interp(grid, unique_values, unique_lambdas, left=np.nan, right=np.nan)


def measure_spectral_enhancement(
    anomaly: xr.DataArray,
    indicator: xr.DataArray,
    events: Sequence[tuple[int, int]],
    strong_fraction: float,
    reference_wavenumbers: tuple[int, int],
    bootstrap_samples: int,
) -> dict[str, object]:
    """Quantify the spectral boost of blocked days relative to calm days.

    The spectral diagnostic works like comparing orchestral recordings:
    blocking days amplify certain harmonics (wavenumbers) so the violin
    section (k=4) stands out against the bass (k=2).
    """

    order_parameter = np.sqrt((anomaly**2).mean(dim=["lat", "lon"]))
    peaks: list[int] = []
    for start, end in events:
        subset = order_parameter.isel(time=slice(start, end))
        if subset.size == 0:
            continue
        local_index = int(np.argmax(subset.values))
        peaks.append(start + local_index)
    if not peaks:
        return {"significant": False, "ratio": np.nan, "confidence_interval": (np.nan, np.nan)}

    count = max(1, int(math.ceil(strong_fraction * len(peaks))))
    peaks = sorted(peaks, key=lambda idx: float(order_parameter.isel(time=idx)))
    strong_days = peaks[-count:]

    block_power = np.mean([_zonal_power(anomaly.isel(time=idx)) for idx in strong_days], axis=0)

    calm_mask = ~indicator.any(dim=["lat", "lon"])
    calm_indices = np.where(calm_mask.values)[0]
    if calm_indices.size == 0:
        return {"significant": False, "ratio": np.nan, "confidence_interval": (np.nan, np.nan)}
    climatology_indices = calm_indices[: min(len(calm_indices), 100)]
    climatology_power = np.mean([
        _zonal_power(anomaly.isel(time=idx)) for idx in climatology_indices
    ], axis=0)

    k_low, k_high = reference_wavenumbers
    ratio_block = block_power[k_high] / (block_power[k_low] + 1e-12)
    ratio_clim = climatology_power[k_high] / (climatology_power[k_low] + 1e-12)
    ratio = ratio_block / (ratio_clim + 1e-12)

    bootstraps = []
    rng = np.random.default_rng(42)
    for _ in range(bootstrap_samples):
        sampled = rng.choice(strong_days, size=len(strong_days), replace=True)
        sample_power = np.mean([_zonal_power(anomaly.isel(time=idx)) for idx in sampled], axis=0)
        boot_ratio_block = sample_power[k_high] / (sample_power[k_low] + 1e-12)
        bootstraps.append(boot_ratio_block / (ratio_clim + 1e-12))
    confidence = (np.percentile(bootstraps, 2.5), np.percentile(bootstraps, 97.5))
    significant = ratio > 1.2 and confidence[0] > 1.0
    return {
        "ratio": ratio,
        "confidence_interval": confidence,
        "significant": bool(significant),
        "block_power": block_power,
        "climatology_power": climatology_power,
    }


def _zonal_power(field: xr.DataArray) -> np.ndarray:
    """Return the zonal power spectrum averaged over latitude."""

    values = field.values
    demeaned = values - values.mean()
    spectrum = np.fft.fft(demeaned, axis=-1)
    return np.mean(np.abs(spectrum) ** 2, axis=-2)


def execute_blocking_pipeline(
    dataset: xr.Dataset,
    output_directory: str | os.PathLike[str],
    config: BlockingConfig | None = None,
) -> dict[str, object]:
    """Run the blocking workflow and emit diagnostics.

    This orchestration function resembles the conductor of an orchestra:
    it cues each section—QGPV, blocking detection, hysteresis, spectra—so
    that the final report harmonizes the disparate measurements.
    """

    cfg = config or BlockingConfig()
    lat_slice = slice(cfg.latitude_band[1], cfg.latitude_band[0])
    ds = dataset.sel(lat=lat_slice)
    z500 = convert_geopotential_to_height(ds["z"] if "z" in ds else ds["z500"])
    qgpv_anomaly = compute_qgpv_anomaly(z500, cfg.reference_latitude)
    indicator = tibaldi_molteni_indicator(
        z500,
        delta_phi=cfg.tm90_delta_phi,
        gradient_threshold=cfg.tm90_gradient_threshold,
        latitude_band=cfg.latitude_band,
    )
    events, filtered_indicator = filter_blocking_events(
        indicator,
        min_duration_days=cfg.min_duration_days,
        min_zonal_extent_deg=cfg.min_zonal_extent_deg,
    )
    lam = evaluate_control_parameter(ds["u"], cfg.latitude_band)
    order_parameter = evaluate_order_parameter(z500, qgpv_anomaly)
    bifurcation = assess_bifurcation(lam, order_parameter, cfg.lambda_bins)
    hysteresis = derive_hysteresis_loop(
        lam,
        order_parameter,
        events,
        cfg.hysteresis_amplitude_grid,
    )
    anomaly = z500.groupby("time.dayofyear") - z500.groupby("time.dayofyear").mean("time")
    spectral = measure_spectral_enhancement(
        anomaly,
        filtered_indicator,
        events,
        cfg.strong_event_fraction,
        cfg.spectral_reference_wavenumbers,
        cfg.bootstrap_samples,
    )

    summary = {
        "bifurcation": bifurcation,
        "hysteresis": hysteresis,
        "spectral": spectral,
        "events": events,
    }

    _render_figures(Path(output_directory), order_parameter, lam, bifurcation, hysteresis, spectral)
    return summary


def _render_figures(
    output_dir: Path,
    order_parameter: xr.DataArray,
    lam: xr.DataArray,
    bifurcation: dict[str, object],
    hysteresis: dict[str, object],
    spectral: dict[str, object],
) -> None:
    """Create summary plots mirroring the analysis briefing."""

    output_dir.mkdir(parents=True, exist_ok=True)

    lam_values = lam.values
    order_values = order_parameter.values
    plt.figure(figsize=(10, 6))
    bins = bifurcation["lambda_bins"]
    legend_needed = False
    for idx in range(len(bins) - 1):
        if idx == len(bins) - 2:
            mask = (lam_values >= bins[idx]) & (lam_values <= bins[idx + 1])
        else:
            mask = (lam_values >= bins[idx]) & (lam_values < bins[idx + 1])
        if mask.sum() < 50:
            continue
        plt.hist(order_values[mask], bins=30, density=True, alpha=0.4, label=f"λ bin {idx + 1}")
        legend_needed = True
    plt.xlabel("Order parameter A")
    plt.ylabel("Probability density")
    plt.title("Bifurcation diagnostic")
    if legend_needed:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "bifurcation_histograms.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(hysteresis["A_grid"], hysteresis["lambda_onset"], label="Onset", linewidth=2)
    plt.plot(hysteresis["A_grid"], hysteresis["lambda_decay"], label="Decay", linewidth=2)
    plt.xlabel("Order parameter A")
    plt.ylabel("λ")
    plt.title("Hysteresis loop")
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "hysteresis_loop.png")
    plt.close()

    if "block_power" in spectral and spectral["block_power"] is not None:
        plt.figure(figsize=(8, 6))
        k = np.arange(len(spectral["block_power"]))
        plt.semilogy(k, spectral["block_power"], label="Blocking peaks")
        plt.semilogy(k, spectral["climatology_power"], label="Calm days")
        plt.xlabel("Zonal wavenumber k")
        plt.ylabel("Power")
        ratio = spectral.get("ratio", float("nan"))
        ci = spectral.get("confidence_interval", (float("nan"), float("nan")))
        plt.title(f"Spectral ratio={ratio:.2f}, CI=({ci[0]:.2f}, {ci[1]:.2f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "spectral_comparison.png")
        plt.close()


__all__ = [
    "BlockingConfig",
    "convert_geopotential_to_height",
    "compute_streamfunction",
    "spherical_laplacian",
    "compute_qgpv_anomaly",
    "tibaldi_molteni_indicator",
    "filter_blocking_events",
    "evaluate_control_parameter",
    "evaluate_order_parameter",
    "assess_bifurcation",
    "derive_hysteresis_loop",
    "measure_spectral_enhancement",
    "execute_blocking_pipeline",
]


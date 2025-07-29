import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Spacetime"))

from src.covariant_resonance import (
    BekensteinParameters,
    f_bekenstein,
    effective_alpha,
    modified_dispersion,
    superradiance_condition,
)
from src.preservation import solve_tau_from_sigma


def test_effective_alpha_scaling():
    params = BekensteinParameters(lambda_b=2.0, alpha_0=1/137)
    phi = 0.1
    eff = effective_alpha(phi, params)
    expected = params.alpha_0 / (1 + params.lambda_b * phi ** 2)
    assert abs(eff - expected) < 1e-12


def test_pce_rescaling():
    params = BekensteinParameters(lambda_b=1.0, pce_sigma=0.5)
    phi = 0.2
    tau_val = solve_tau_from_sigma(params.pce_sigma)
    assert f_bekenstein(phi, params) == params.lambda_b * phi ** 2 * tau_val


def test_modified_dispersion_basic():
    params = BekensteinParameters(lambda_b=1.0)
    omega_sq = modified_dispersion(k=1.0, phi=0.0, phi_prime=0.0, params=params)
    assert abs(omega_sq - 2.0) < 1e-12


def test_superradiance_condition():
    assert superradiance_condition(omega=0.9, m=1, omega_h=1.0)
    assert not superradiance_condition(omega=1.1, m=1, omega_h=1.0)


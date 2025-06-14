"""
Spacetime Recursion Core Engine
"""

from src.preservation import check_preservation
from src.torsion import model_torsion_flow
from src.spinor import simulate_spinor_field
from src.lattice import detect_lambda_4_eigenmode


def run_spacetime_engine():
    print("🚀 Running the Spacetime Recursion Engine...")

    sigma = 1.0
    tau, ok = check_preservation(sigma)
    print(f"PCE satisfied? {'Yes' if ok else 'No'} (tau={tau:.3f})")

    spin_density = 0.75
    beta = 1.2
    beta_tilde = 0.9
    torsion_force = model_torsion_flow(spin_density, beta, beta_tilde, lambda r: r**2)
    print(f"Torsion Field Output: {torsion_force}")

    signal = [4, 4, 2, 3]
    is_lambda_4 = detect_lambda_4_eigenmode(signal)
    print(f"λ = 4 eigenmode detected? {'Yes' if is_lambda_4 else 'No'}")

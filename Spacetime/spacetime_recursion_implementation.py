"""
Spacetime Recursion Core Engine
"""

from src.preservation import evaluate_preservation_constraint
from src.torsion import model_torsion_flow
from src.spinor import simulate_spinor_field
from src.lattice import detect_lambda_4_eigenmode


def run_spacetime_engine():
    print("🚀 Running the Spacetime Recursion Engine...")

    sigma = 1.0
    tau = 1.0
    upsilon = 1.0

    constraint = evaluate_preservation_constraint(sigma, tau, upsilon)
    print(f"Preservation Constraint Output: {constraint}")

    spin_density = 0.75
    beta = 1.2
    beta_tilde = 0.9
    torsion_force = model_torsion_flow(spin_density, beta, beta_tilde, lambda r: r**2)
    print(f"Torsion Field Output: {torsion_force}")

    signal = [4, 4, 2, 3]
    is_lambda_4 = detect_lambda_4_eigenmode(signal)
    print(f"λ = 4 eigenmode detected? {'Yes' if is_lambda_4 else 'No'}")

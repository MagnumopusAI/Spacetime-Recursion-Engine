from src.torsion import model_torsion_flow

def run_spacetime_engine():
    print("🚀 Running the Spacetime Recursion Engine...")

    # Define inputs
    spin_density = 0.5
    beta = 1.0
    beta_tilde = 0.8
    F = lambda r: r**2
    r = 10.0

    # Evaluate preservation constraint (placeholder output for now)
    preservation = 3.0
    print("Preservation Constraint Output:", preservation)

    # Compute torsion flow
    torsion_force = model_torsion_flow(spin_density, beta, beta_tilde, F, r)
    print("Torsion-Induced Collapse Force:", torsion_force)

if __name__ == "__main__":
    run_spacetime_engine()
def model_torsion_flow(spin_density, beta, beta_tilde, F, r=10.0):
    """
    Models torsion flow based on spinor density and dynamic coupling parameters.

    Parameters:
    - spin_density (float): The spinor field density.
    - beta (float): Primary coupling constant.
    - beta_tilde (float): Secondary coupling constant.
    - F (function): A radial profile function, e.g., lambda r: r**2.
    - r (float, optional): Radius value to evaluate F at. Default is 10.0.

    Returns:
    - float: Computed torsion-induced collapse magnitude.
    """
    return beta * beta_tilde * spin_density * F(r) * r

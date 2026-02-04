import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for automation

# Ensure we can import the local package
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig
from cubed_sphere.utils import plot_cubed_sphere_state

def main():
    # Configuration
    backend = os.environ.get('CUBED_SPHERE_BACKEND', 'numpy')
    print(f"Running Advection with backend: {backend}")
    config = AdvectionConfig(
        N=24,
        R=1.0,
        u0=2 * np.pi,
        alpha0=0.0,
        CFL=1.0,
        T_final=1.0,
        backend=backend
    )
    
    print("Configuration:", config)
    
    # Initialize Solver (Stateless Physics Definition)
    solver = CubedSphereAdvectionSolver(config)
    
    # Generate Initial State separately (Data)
    initial_state = solver.get_initial_condition(type="gaussian")
    
    # Run Simulation (Solver applies Physics to Data)
    # Solve from t=0.0 to t=T_final
    final_state = solver.solve((0.0, config.T_final), initial_state)
    
    # Visualization
    plot_cubed_sphere_state(solver, final_state, title=f"Advection Result (N={config.N})")

if __name__ == "__main__":
    main()

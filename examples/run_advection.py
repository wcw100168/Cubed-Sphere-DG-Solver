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
    
    # 1. Physics Parameters
    u_max = 2 * np.pi # Maximum velocity (Solid Body Rotation)
    N = 24
    target_cfl = 1.0 # Advection can often handle CFL~1 with RK strong stability
    
    # 2. Dynamic Time Step (Stability ~ 1/N^2)
    # dt = CFL / (v_max * N^2)
    dt = target_cfl / (u_max * N**2)
    print(f"Computed dt: {dt:.5f} (N={N}, u_max={u_max:.2f}, CFL={target_cfl})")
    
    config = AdvectionConfig(
        N=N,
        R=1.0,
        u0=u_max,
        alpha0=0.0,
        CFL=target_cfl,
        dt=dt, # Explicitly valid stability condition
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

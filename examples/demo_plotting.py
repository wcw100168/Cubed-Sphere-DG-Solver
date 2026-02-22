"""
Demo script for visualizing Cubed Sphere data using global projections (Mollweide).
Requirements: matplotlib, numpy
"""
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig
from cubed_sphere.utils.vis import plot_cubed_sphere_state

def main():
    print("--- Running Cubed Sphere Plotting Demo ---")
    
    # 1. Initialize Grid (via Solver)
    # Using N=32 for decent resolution
    config = AdvectionConfig(N=32, backend='numpy')
    solver = CubedSphereAdvectionSolver(config)
    
    # 2. Generate Data (Twin Gaussian)
    print("Generating synthetic data...")
    def twin_gaussian(lon, lat):
        # Peak 1 at (0, 0)
        d1 = np.arccos(np.cos(lat) * np.cos(lon))
        val1 = np.exp(-(d1 / 0.5)**2)
        # Peak 2 at (pi, 0)
        d2 = np.arccos(np.cos(lat) * np.cos(lon - np.pi))
        val2 = np.exp(-(d2 / 0.5)**2)
        return val1 + val2

    u0 = solver.get_initial_condition(type="custom", func=twin_gaussian)
    
    # 3. Plotting
    # We use a Mollweide projection which unwraps the sphere.
    output_file = "demo_plot_mollweide.png"
    
    title = "Twin Gaussian Initial Condition (N=32)"
    print(f"Plotting '{title}' to {output_file}...")
    
    # Note: plot_cubed_sphere_state expects state of shape (6, N, N)
    # If u0 is (3, 6, N, N) from SWE, pass u0[0]
    
    try:
        plot_cubed_sphere_state(solver, u0, title=title, save_path=output_file)
        print(f"✅ Saved {output_file}")
    except Exception as e:
        print(f"❌ Plotting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

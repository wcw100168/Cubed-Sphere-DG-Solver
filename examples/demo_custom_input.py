
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig
from cubed_sphere.solvers.swe import CubedSphereSWE, SWEConfig
from cubed_sphere.utils.ic_builder import build_from_function
from cubed_sphere.geometry.grid import CubedSphereEquiangular, lgl_nodes_weights

def demo_functional_init(solver_type="advection"):
    print(f"\n--- Running Functional Initialization Demo ({solver_type}) ---")
    
    # Define a custom function: Twin Gaussian
    def twin_gaussian(lon, lat):
        # Peak 1 at (0, 0)
        d1 = np.arccos(np.cos(lat) * np.cos(lon))
        val1 = np.exp(-(d1 / 0.5)**2)
        
        # Peak 2 at (pi, 0)
        d2 = np.arccos(np.cos(lat) * np.cos(lon - np.pi))
        val2 = np.exp(-(d2 / 0.5)**2)
        
        return val1 + val2

    if solver_type == "advection":
        config = AdvectionConfig(N=32, CFL=1.0, T_final=0.0, backend='numpy')
        solver = CubedSphereAdvectionSolver(config)
        u0 = build_from_function(solver, twin_gaussian)
        plot_data = u0[0] # Face 1
        
    elif solver_type == "swe":
        config = SWEConfig(N=8, T_final=0.0, backend='numpy')
        solver = CubedSphereSWE(config)

        # Apply Gaussian to mass variable (var_idx=0) and include sqrt_g
        u0 = build_from_function(
            solver,
            twin_gaussian,
            var_idx=0,
            multiply_by_sqrt_g=True,
        )
        plot_data = u0[0, 0] / solver._impl.faces[solver._impl.topology.FACE_MAP[0]].sqrt_g
    
    print(f"Initialized state with min={u0.min():.4f}, max={u0.max():.4f}")
    
    # Simple visualization stats
    # Check shape to decide how to print mean
    if u0.ndim == 3: # Advection (6, N, N)
         print(f"Face 1 Mean: {u0[0].mean():.4f}")
    elif u0.ndim == 4: # SWE (3, 6, N, N)
         print(f"Face 1 Mean (Var 0): {u0[0, 0].mean():.4f}")

    # Plot (quick contour of one face if matplotlib is available)
    try:
        plt.figure()
        plt.imshow(plot_data)
        plt.title(f"Face 1 - Twin Gaussian ({solver_type})")
        plt.colorbar()
        plt.savefig(f"demo_functional_init_{solver_type}.png")
        print(f"Saved plot to demo_functional_init_{solver_type}.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", type=str, default="advection", choices=["advection", "swe"])
    args = parser.parse_args()
    
    demo_functional_init(args.solver)

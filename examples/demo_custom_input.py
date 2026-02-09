
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig
from cubed_sphere.solvers.swe import CubedSphereSWE, SWEConfig
from cubed_sphere.utils import regrid
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
        u0 = solver.get_initial_condition(type="custom", func=twin_gaussian)
        plot_data = u0[0] # Face 1
        
    elif solver_type == "swe":
        config = SWEConfig(N=8, T_final=0.0, backend='numpy')
        solver = CubedSphereSWE(config)
        
        # Manual Initialization for System of Equations
        # We need to construct state (3, 6, N+1, N+1)
        num_nodes = config.N + 1
        u0 = np.zeros((3, 6, num_nodes, num_nodes))
        
        # Apply Gaussian to Height (Var 0)
        # We need to iterate faces to get lat/lon for the function
        topo = solver._impl.topology
        faces = solver._impl.faces
        
        for i, fname in enumerate(topo.FACE_MAP):
            fg = faces[fname]
            # Eval function
            h_val = twin_gaussian(fg.lon, fg.lat)
            # Multiply by sqrt_g (Mass variable)
            u0[0, i] = h_val * fg.sqrt_g
            # u0[1] and u0[2] remain zero (no velocity)
            
        plot_data = u0[0, 0] / faces[topo.FACE_MAP[0]].sqrt_g # Plot Height on Face 1
    
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

def demo_regrid_init():
    # Only implemented for Advection in this demo for simplicity
    print("\n--- Running Regridding Initialization Demo (Advection Only) ---")
    
    config = AdvectionConfig(N=32, T_final=0.0)
    solver = CubedSphereAdvectionSolver(config)
    
    # 1. Create Mock Weather Data (Lat/Lon Grid)
    n_lat, n_lon = 90, 180
    lat_1d = np.linspace(-90, 90, n_lat)
    lon_1d = np.linspace(-180, 180, n_lon) # -180 to 180 convention
    
    # Create valid meshgrid for data generation
    # Note: meshgrid default is xy, meaning (lat, lon) access needs care if not 'ij'
    LON, LAT = np.meshgrid(lon_1d, lat_1d) 
    
    # Pattern: Zonal waves
    weather_data = np.cos(np.deg2rad(LAT)) * np.sin(3 * np.deg2rad(LON))
    
    print(f"Mock Data shape: {weather_data.shape} (Lat x Lon)")
    
    # 2. Regrid to Cubed Sphere
    state = np.zeros((6, config.N, config.N))
    
    print("Regridding...")
    for i, fname in enumerate(solver.topology.FACE_MAP):
        fg = solver.faces[fname]
        state[i] = regrid.from_latlon_grid(fg, lat_1d, lon_1d, weather_data)
        
    print(f"Regridded state with min={state.min():.4f}, max={state.max():.4f}")

    # Plot
    try:
        plt.figure()
        plt.imshow(state[0])
        plt.title("Face 1 - Regridded Zonal Waves")
        plt.colorbar()
        plt.savefig("demo_regrid_init.png")
        print("Saved plot to demo_regrid_init.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", type=str, default="advection", choices=["advection", "swe"])
    args = parser.parse_args()
    
    demo_functional_init(args.solver)
    if args.solver == "advection":
        demo_regrid_init()

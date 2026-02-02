
import numpy as np
import matplotlib.pyplot as plt
from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig
from cubed_sphere.utils import regrid
from cubed_sphere.geometry.grid import CubedSphereEquiangular, lgl_nodes_weights

def demo_functional_init():
    print("\n--- Running Functional Initialization Demo ---")
    
    config = AdvectionConfig(N=32, CFL=1.0, T_final=0.0, backend='numpy')
    solver = CubedSphereAdvectionSolver(config)
    
    # Define a custom function: Twin Gaussian
    def twin_gaussian(lon, lat):
        # Peak 1 at (0, 0)
        d1 = np.arccos(np.cos(lat) * np.cos(lon))
        val1 = np.exp(-(d1 / 0.5)**2)
        
        # Peak 2 at (pi, 0)
        d2 = np.arccos(np.cos(lat) * np.cos(lon - np.pi))
        val2 = np.exp(-(d2 / 0.5)**2)
        
        return val1 + val2

    # Initialize
    u0 = solver.get_initial_condition(type="custom", func=twin_gaussian)
    print(f"Initialized state with min={u0.min():.4f}, max={u0.max():.4f}")
    
    # Simple visualization stats
    for i in range(6):
        print(f"Face {i+1} Mean: {u0[i].mean():.4f}")

    # Plot (quick contour of one face if matplotlib is available)
    try:
        plt.figure()
        plt.imshow(u0[0])
        plt.title("Face 1 - Twin Gaussian")
        plt.colorbar()
        plt.savefig("demo_functional_init.png")
        print("Saved plot to demo_functional_init.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

def demo_regrid_init():
    print("\n--- Running Regridding Initialization Demo ---")
    
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
    demo_functional_init()
    demo_regrid_init()

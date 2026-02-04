import numpy as np
import os
import sys
import matplotlib
matplotlib.use("Agg") # Non-interactive backend

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig
from cubed_sphere.utils.io import NetCDFMonitor
from cubed_sphere.utils.vis import animate_from_netcdf

def main():
    # 1. Setup Simulation
    # Use N=16 for speed in demo
    config = AdvectionConfig(
        N=16,
        R=1.0,
        u0=2.0 * np.pi,
        CFL=0.5,
        T_final=0.2, # Short run for demo
        backend='numpy'
    )
    
    solver = CubedSphereAdvectionSolver(config)
    u0 = solver.get_initial_condition(type="gaussian")
    
    # 2. Attach Monitor
    filename = "simulation_data.nc"
    # Save frequently to get frames
    monitor = NetCDFMonitor(filename, save_interval=0.02)
    
    # 3. Run with Callback
    print("Running simulation with I/O monitor...")
    solver.solve((0.0, config.T_final), u0, callbacks=[monitor])
    
    # 4. Render Offline
    output_mp4 = "advection_movie.mp4"
    if os.path.exists(output_mp4):
        os.remove(output_mp4)
        
    print(f"Rendering animation from {filename}...")
    try:
        animate_from_netcdf(filename, output_mp4, fps=5)
        print(f"Animation saved to {output_mp4}")
    except Exception as e:
        print(f"Animation failed: {e}")
        # Could be ffmpeg missing, but xarray/matplotlib logic should be tested
    
    print("Demo Complete.")
    
if __name__ == "__main__":
    main()

import numpy as np
import os
import sys
import matplotlib
matplotlib.use("Agg") # Non-interactive backend

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig
from cubed_sphere.solvers.swe import CubedSphereSWE, SWEConfig
from cubed_sphere.utils.io import NetCDFMonitor
from cubed_sphere.utils.vis import animate_from_netcdf

def main():
    parser = argparse.ArgumentParser(description="Demo Offline I/O and Animation")
    parser.add_argument("--solver", type=str, default="advection", choices=["advection", "swe"], help="Solver type")
    args = parser.parse_args()

    # 1. Setup Simulation
    if args.solver == "advection":
        print("Initializing Advection Solver...")
        # Auto-compute dt
        N = 16
        R = 1.0
        u0 = 2.0 * np.pi
        target_cfl = 0.5
        dt = target_cfl * R / (u0 * N**2) # Stability Condition
        
        config = AdvectionConfig(
            N=N, R=R, u0=u0, CFL=target_cfl,
            T_final=0.2, dt=dt, backend='numpy'
        )
        solver = CubedSphereAdvectionSolver(config)
        u0_state = solver.get_initial_condition(type="gaussian")
        var_name_plot = "Concentration"
        
    elif args.solver == "swe":
        print("Initializing SWE Solver...")
        # Auto-compute dt
        N = 8
        H = 10000.0
        g = 9.80616
        R = 6.371e6
        c_wave = np.sqrt(g * H)
        v_flow = 40.0 # approx
        v_max = c_wave + v_flow
        target_cfl = 0.5
        dt = target_cfl * R / (v_max * N**2)
        
        config = SWEConfig(
            N=N, T_final=0.1, dt=dt, backend='numpy'
        )
        solver = CubedSphereSWE(config)
        u0_state = solver.get_initial_condition(type="case2")
        var_name_plot = "Fluid Depth (h)"
    
    # 2. Attach Monitor
    filename = "simulation_data.nc"
    # Save frequently to get frames
    monitor = NetCDFMonitor(filename, save_interval=dt*10)
    
    # 3. Run with Callback
    print(f"Running {args.solver} simulation with I/O monitor (dt={dt:.4e})...")
    solver.solve((0.0, config.T_final), u0_state, callbacks=[monitor])
    
    # 4. Render Offline
    output_mp4 = f"{args.solver}_movie.mp4"
    if os.path.exists(output_mp4):
        os.remove(output_mp4)
        
    print(f"Rendering animation from {filename}...")
    try:
        # For SWE, var_idx=0 is h (Height). For Advection, var_idx=0 is the only variable.
        print(f"Plotting variable index 0: {var_name_plot}")
        animate_from_netcdf(filename, output_mp4, fps=5, var_idx=0)
        print(f"Animation saved to {output_mp4}")
    except Exception as e:
        print(f"Animation failed: {e}")
        # Could be ffmpeg missing, but xarray/matplotlib logic should be tested
    
    print("Demo Complete.")
    
if __name__ == "__main__":
    main()

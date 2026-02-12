"""
Williamson Case 6: Rossby-Haurwitz Wave
---------------------------------------
Validation Step 4: Tests Non-Linear Stability and Wave Propagation.

Physics:
- Wavenumber-4 planetary wave.
- Initial state is a solution to the NON-DIVERGENT Barotropic Vorticity Equation.
- It is NOT an exact solution to the full Shallow Water Equations.
- The wave moves eastward with angular velocity approx 'omega' (but changes shape).
- A rigorous test for numerical stability and dispersion relations.

Ref:
Williamson et al. (1992). Case 6.
"""

import numpy as np
import argparse
import sys
import os
import time
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from cubed_sphere.solvers import CubedSphereSWE, SWEConfig

def main():
    parser = argparse.ArgumentParser(description="Williamson Case 6: Rossby-Haurwitz Wave")
    parser.add_argument("--N", type=int, default=32, help="Polynomial order")
    parser.add_argument("--days", type=float, default=14.0, help="Simulation duration in days")
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "jax"])
    parser.add_argument("--dt", type=float, default=30.0, help="Time step (s)")
    parser.add_argument("--output", type=str, default="rossby_wave_comparison.png", help="Output plot file")
    args = parser.parse_args()

    # Constants
    R = 6.37122e6
    Omega = 7.292e-5
    g = 9.80616
    h0 = 8000.0
    
    # Case 6 Parameters
    omega_wave = 7.848e-6
    K = 7.848e-6
    R_wave = 4.0

    print(f"--- Williamson Case 6 (Rossby-Haurwitz Wave) ---")
    print(f"Backend: {args.backend}")
    print(f"Resolution N: {args.N}")
    print(f"Duration: {args.days} days")
    print(f"Time Step: {args.dt} s")

    # Configure Solver
    config = SWEConfig(
        N=args.N,
        R=R,
        Omega=Omega,
        gravity=g,
        H_avg=h0,
        backend=args.backend,
        dt=args.dt,
        filter_order=16 
    )
    
    solver = CubedSphereSWE(config)
    
    # --- Initialization ---
    print("Generating Initial Conditions (Case 6)...")
    try:
        # Pass params to get_initial_condition
        state_init = solver.get_initial_condition(
            type="case6", 
            omega=omega_wave, 
            K=K, 
            R_wave=R_wave, 
            h0=h0
        )
    except Exception as e:
        print(f"Error initializing Case 6: {e}")
        sys.exit(1)

    # --- Run Simulation ---
    print("Starting integration...")
    T_total = args.days * 24.0 * 3600.0
    
    solver.state = state_init.copy()
    
    t0 = time.time()
    
    # Check mass conservation during run? 
    # Just run solve directly.
    state_final = solver.solve((0.0, T_total), state_init)
    
    t1 = time.time()
    print(f"Integration complete in {t1-t0:.2f}s")
    
    # --- Analysis ---
    
    # 1. Mass Conservation
    topo = solver._impl.topology
    faces = solver._impl.faces
    
    def compute_mass(state):
        mass = 0.0
        for i, fname in enumerate(topo.FACE_MAP):
            fg = faces[fname]
            if hasattr(fg, 'walpha'):
                 w = fg.walpha
            elif hasattr(fg, 'w'):
                 w = fg.w
            else:
                 # Fallback: create LGL weights if missing (should be in grid.py)
                 w = np.zeros(args.N+1) # Panic?
                 pass
            
            # W2D
            W2D = np.outer(w, w)
            # state[0] is h * sqrt_g (Mass Density in Computational Space)
            # Integral = Sum (state[0] * w * w)
            # Note: The Jacobian sqrt_g is ALREADY in state[0].
            # So we just sum state[0] * weights.
            mass += np.sum(state[0, i] * W2D)
        return mass
        
    m0 = compute_mass(state_init)
    m_final = compute_mass(state_final)
    m_err = (m_final - m0) / m0
    
    print(f"\n--- Results ---")
    print(f"Initial Mass: {m0:.6e}")
    print(f"Final Mass:   {m_final:.6e}")
    print(f"Mass Rel Err: {m_err:.6e}")
    
    if abs(m_err) < 1e-11:
        print("PASS: Mass Conservation Excellent")
    elif abs(m_err) < 1e-9:
        print("PASS: Mass Conservation Acceptable")
    else:
        print("WARNING: Significant Mass Drift")

    # 2. Plotting (Equatorial Cross Section or Full Map?)
    # Since cubedsphere is hard to plot 2D simply without cartopy/regrid,
    # We will plot the Height Field along the Equator (approx).
    # Face 1 (East) and Face 3 (West), Face 4/5 (North/South).
    # Actually Face 0, 1, 2, 3 are equatorial band usually.
    # panel 0: (-x), 1: (+x), 2: (-y), 3: (+y) ?
    # Let's check Topology.
    # West(0), East(1), South(2), North(3), Bottom(4), Top(5)?
    # Just plot statistics or select points.
    
    # Let's regrid to Lat-Lon for plotting using `utils.vis` or manual.
    # Or just flatten all data and plot vs Longitude for points near equator (Lat ~ 0).
    
    print(f"Generating plot: {args.output}")
    
    lons_all = []
    lats_all = []
    h_init_all = []
    h_final_all = []
    
    for i, fname in enumerate(topo.FACE_MAP):
        fg = faces[fname]
        # Coordinates
        lam = fg.lon.flatten()
        th = fg.lat.flatten()
        
        # Height
        h_in = (state_init[0, i] / fg.sqrt_g).flatten()
        h_fi = (state_final[0, i] / fg.sqrt_g).flatten()
        
        lons_all.append(lam)
        lats_all.append(th)
        h_init_all.append(h_in)
        h_final_all.append(h_fi)
        
    lons_all = np.concatenate(lons_all)
    lats_all = np.concatenate(lats_all)
    h_init_all = np.concatenate(h_init_all)
    h_final_all = np.concatenate(h_final_all)
    
    print(f"\n--- Field Statistics ---")
    print(f"Height Range Day 0:  [{np.min(h_init_all):.4f}, {np.max(h_init_all):.4f}] m")
    print(f"Height Range Day {int(args.days)}: [{np.min(h_final_all):.4f}, {np.max(h_final_all):.4f}] m")
    
    # Filter for Equator (+- 2 degrees)
    mask = np.abs(lats_all) < np.deg2rad(5.0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(np.rad2deg(lons_all[mask]), h_init_all[mask], s=1, label="Day 0", alpha=0.5)
    ax.scatter(np.rad2deg(lons_all[mask]), h_final_all[mask], s=1, label=f"Day {int(args.days)}", alpha=0.5)
    
    ax.set_title(f"Williamson Case 6: Height Field at Equator (N={args.N})")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Geopotential Height (m) / Height (m)")
    ax.legend()
    
    plt.savefig(args.output)
    print("Plot saved.")
    
    # 3. Stability Check
    if np.any(np.isnan(state_final)):
        print("FAIL: Simulation Exploded (NaNs detected)")
    else:
        print("PASS: Simulation Stable")


if __name__ == "__main__":
    main()

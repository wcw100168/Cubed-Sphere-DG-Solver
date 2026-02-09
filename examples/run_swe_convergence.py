import argparse
import numpy as np
import time
import sys
import os

# Enable JAX x64 (Must be before any JAX import occurs indirectly)
import os
os.environ["JAX_PLATFORMS"] = "cpu" # Default to CPU for stability
try:
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

# Add project root to path if running from examples/
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cubed_sphere.solvers.swe import CubedSphereSWE, SWEConfig

def setup_case2_initial_condition(solver, config):
    """
    Sets up Williamson Case 2 (Global Steady State Zonal Flow).
    Returns the initial state array (3, 6, N+1, N+1).
    """
    # Grid size
    num_nodes = config.N + 1
    # Use Var-Major layout: (Vars, Face, Xi, Eta)
    state = np.zeros((3, 6, num_nodes, num_nodes))
    
    # Physics Parameters
    u0 = 2.0 * np.pi * config.R / (12.0 * 24.0 * 3600.0) # ~38 m/s
    h0 = config.H_avg # Base depth
    g = config.gravity
    R = config.R
    Omega = config.Omega
    
    # Access Topology and Faces depending on backend
    # The Facade hides implementation, but we need access to geometry for IC.
    # We access `solver._impl.faces` which works for both NumPy and JAX backends
    # (provided we handle JAX arrays if needed).
    
    # We will compute everything in NumPy first.
    
    topo = solver._impl.topology
    faces = solver._impl.faces
    
    for i, fname in enumerate(topo.FACE_MAP):
        fg = faces[fname]
        
        # Get coordinates (NumPy extraction if JAX)
        if hasattr(fg.lat, 'device') or hasattr(fg.lat, 'sharding'): # JAX array check
            lat = np.array(fg.lat)
            lon = np.array(fg.lon)
            g1_vec = np.array(fg.g1_vec)
            g2_vec = np.array(fg.g2_vec)
            g_inv = np.array(fg.g_inv)
        else:
            lat = fg.lat
            lon = fg.lon
            g1_vec = fg.g1_vec
            g2_vec = fg.g2_vec
            g_inv = fg.g_inv
            
        # Analytical Solution (Spherical)
        u_sph = u0 * np.cos(lat)
        v_sph = 0.0
        h_val = h0 - (R * Omega * u0 + 0.5 * u0**2) * (np.sin(lat)**2) / g
        
        # Convert Vector to Contravariant (u1, u2)
        # Cartesian Basis Vectors at (lon, lat)
        # e_lambda = (-sin(lon), cos(lon), 0)
        # e_theta = (-sin(lat)cos(lon), -sin(lat)sin(lon), cos(lat))
        
        e_lam_x = -np.sin(lon)
        e_lam_y = np.cos(lon)
        e_lam_z = np.zeros_like(lon)
        
        e_th_x = -np.sin(lat)*np.cos(lon)
        e_th_y = -np.sin(lat)*np.sin(lon)
        e_th_z = np.cos(lat)
        
        # Velocity Vector V in Cartesian
        # V = u_sph * e_lambda + v_sph * e_theta
        # Since v_sph = 0 for Case 2:
        Vx = u_sph * e_lam_x
        Vy = u_sph * e_lam_y
        Vz = u_sph * e_lam_z
        
        # Project onto Covariant Basis (u_1, u_2)
        # The Solver (Vector Invariant) expects COVARIANT components.
        # u_i = V . g_i
        V_dot_g1 = Vx * g1_vec[..., 0] + Vy * g1_vec[..., 1] + Vz * g1_vec[..., 2]
        V_dot_g2 = Vx * g2_vec[..., 0] + Vy * g2_vec[..., 1] + Vz * g2_vec[..., 2]
        
        # State[0] is Mass = sqrt_g * h
        # Access sqrt_g carefully
        if hasattr(fg.sqrt_g, 'device') or hasattr(fg.sqrt_g, 'sharding'):
             sqrt_g = np.array(fg.sqrt_g)
        else:
             sqrt_g = fg.sqrt_g
             
        state[0, i] = h_val * sqrt_g
        state[1, i] = V_dot_g1
        state[2, i] = V_dot_g2
        
    return state

def compute_errors(final_state, exact_state, solver):
    """Computes L2 and Linf errors for the height field h."""
    diff_h_all = []
    
    topo = solver._impl.topology
    faces = solver._impl.faces
    
    for i, fname in enumerate(topo.FACE_MAP):
        fg = faces[fname]
        if hasattr(fg.sqrt_g, 'device') or hasattr(fg.sqrt_g, 'sharding'):
             sqrt_g = np.array(fg.sqrt_g)
        else:
             sqrt_g = fg.sqrt_g
             
        # Use Var-Major indexing [Var, Face]
        h_final = final_state[0, i] / sqrt_g
        h_exact = exact_state[0, i] / sqrt_g
        
        diff_h_all.append(np.ravel(h_final - h_exact))
        
    diff_h_flat = np.concatenate(diff_h_all)
    
    err_inf = np.max(np.abs(diff_h_flat))
    err_l2 = np.sqrt(np.mean(diff_h_flat**2))
    
    return err_l2, err_inf

def main():
    parser = argparse.ArgumentParser(description="SWE Solver Spectral Convergence Test")
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "jax"], help="Solver backend")
    parser.add_argument("--hours", type=float, default=2.0, help="Simulation time in hours")
    parser.add_argument("--min_n", type=int, default=8, help="Minimum N (polynomial order)")
    parser.add_argument("--max_n", type=int, default=32, help="Maximum N")
    args = parser.parse_args()
    
    # JAX Configuration
    if args.backend == "jax":
        import jax
        jax.config.update("jax_enable_x64", True)
        # Optional: Set platform to CPU if Metal is unstable
        # os.environ["JAX_PLATFORMS"] = "cpu"
        print("JAX Backend Enabled (x64 Precision)")

    T_final = args.hours * 3600.0
    N_values = list(range(args.min_n, args.max_n + 1, 4)) # Step 4 usually good for spectral
    if len(N_values) < 2:
        N_values = list(range(args.min_n, args.max_n + 1, 2))
        
    results = []
    
    print(f"\nExample: SWE Geometric Convergence Test (Case 2)")
    print(f"Backend: {args.backend}")
    print(f"Time: {args.hours} hours ({T_final} s)")
    print(f"{'-'*65}")
    print(f"{'N':<5} | {'dt (s)':<10} | {'L2 Error':<12} | {'Linf Error':<12} | {'Time (s)':<8}")
    print(f"{'-'*65}")
    
    state_exact = None
    
    for N in N_values:
        # Dynamic Time Step Scaling for DG (Stability ~ 1/N^2)
        # 1. Estimate Wave Speed
        # Case 2: u ~ 30-40 m/s, c = sqrt(gH) ~ 313 m/s
        # v_max = |u| + c
        u_max = 2.0 * np.pi * 6.37122e6 / (12.0 * 24.0 * 3600.0) # ~ 38 m/s
        c_wave = np.sqrt(9.80616 * 10000.0) # ~ 313 m/s
        v_total = u_max + c_wave
        
        # 2. Compute dt based on CFL target
        target_cfl = 0.5
        # Note: Cubed Sphere element size scales with R/N. 
        # But polynomial stiffness scales ~ N^2.
        # dt ~ CFL * (Scale_Factor) / (v_max * N^2) usually.
        # For simplicity here, we stick to the user mandate: dt = CFL / (v_max * N^2) * Scaling? 
        # Actually, standard formula often includes Radius.
        # dt = CFL * (R / N^2) / v_max ? 
        # Let's use the provided User Formula: dt = CFL / (v_max * N^2) 
        # BEWARE: This might be excessively small if "1" is not scaled by R.
        # If v_max has units m/s, N is dimensionless. "1" needs length units?
        # Re-reading prompt: "dt = CFL / (v_max * N^2)"
        # If this results in 1e-9, simulations won't finish. 
        # Context: Previous dt was 10.0 * (8/N)^2.
        # Let's adapt the heuristic to be Dimensionally Consistent or relative to grid.
        # Grid spacing dx ~ R / N.
        # CFL = v_max * dt / (dx / N) ~ v_max * dt * N / (R/N) = v * dt * N^2 / R
        # => dt = CFL * R / (v_max * N^2).
        
        R_earth = 6.37122e6
        dt = target_cfl * R_earth / (v_total * N**2)
        
        config = SWEConfig(
            N=N,
            R=R_earth,
            Omega=7.292e-5,
            gravity=9.80616,
            H_avg=10000.0,
            backend=args.backend,
            filter_order=16, # Fixed filter order
            dt=dt
        )
        
        solver = CubedSphereSWE(config)
        
        # Setup Initial Condition
        initial_state = setup_case2_initial_condition(solver, config)
        
        # Run
        start_time = time.time()
        final_state = solver.solve((0.0, T_final), initial_state)
        # Ensure numpy array
        final_state = np.array(final_state)
        wall_time = time.time() - start_time
        
        # Compute Error
        # Exact solution for Case 2 is the initial condition itself
        err_l2, err_inf = compute_errors(final_state, initial_state, solver)
        
        results.append({
            "N": N,
            "L2": err_l2,
            "Linf": err_inf,
            "dt": dt,
            "time": wall_time
        })
        
        print(f"{N:<5} | {dt:<10.5f} | {err_l2:<12.5e} | {err_inf:<12.5e} | {wall_time:<8.2f}")

    # Print Final Markdown Table with Rates
    print("\n\n### Convergence Results Table")
    print(f"| N | L2 Error | Linf Error | L2 Rate | Linf Rate |")
    print(f"|---|---|---|---|---|")
    
    for i, res in enumerate(results):
        N = res["N"]
        l2 = res["L2"]
        linf = res["Linf"]
        
        if i > 0:
            prev = results[i-1]
            N_prev = prev["N"]
            # Algebraic convergence rate: log(E_prev / E) / log(N / N_prev)
            # Note: Slope in log-log plot. k = -log(y2/y1)/log(x2/x1) typically.
            # Rate = log(E1/E2) / log(N2/N1)
            if l2 > 0 and prev["L2"] > 0:
                rate_l2 = np.log(prev["L2"] / l2) / np.log(N / N_prev)
                r_l2_str = f"{rate_l2:.2f}"
            else:
                r_l2_str = "Inf" if prev["L2"] > 0 else "NaN"

            if linf > 0 and prev["Linf"] > 0:
                rate_linf = np.log(prev["Linf"] / linf) / np.log(N / N_prev)
                r_linf_str = f"{rate_linf:.2f}"
            else:
                 r_linf_str = "Inf" if prev["Linf"] > 0 else "NaN"
        else:
            r_l2_str = "-"
            r_linf_str = "-"
            
        print(f"| {N} | {l2:.4e} | {linf:.4e} | {r_l2_str} | {r_linf_str} |")

if __name__ == "__main__":
    main()

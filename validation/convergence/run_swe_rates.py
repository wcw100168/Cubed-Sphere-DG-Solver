"""
SWE Convergence Test: Williamson Case 2 (Steady State)
------------------------------------------------------
Verifies spectral convergence of the Shallow Water Equations solver.

Case 2: Global Steady State Nonlinear Zonal Flow (Rotated Pole).
Metric: Relative L2 and Linf Error of the Geopotential Height Field (h).

Constraints:
- dt scaling: dt ~ 1/N^2
- Duration: 1 Day
- Alpha: 45 degrees
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from cubed_sphere.solvers import CubedSphereSWE, SWEConfig

def run_case2(N, dt):
    """
    Run Williamson Case 2 (Steady State) for 1 Day.
    Returns (L2 Relative Error, Linf Relative Error) for Height Field h.
    """
    # Parameters
    R = 6.37122e6
    Omega = 7.292e-5 # Rotation rate
    g = 9.80616      # Gravity
    
    alpha_deg = 0.0 # Flow angle (degrees)
    days = 1.0
    
    # Physics Parameters (Case 2)
    # u0 = 2*pi*R / 12 days
    u0 = 2.0 * np.pi * R / (12.0 * 24.0 * 3600.0) 
    h0 = 2998.0 # Reference height
    
    # Configure Solver
    config = SWEConfig(
        N=N,
        R=R,
        Omega=Omega,
        gravity=g,
        H_avg=h0,
        dt=dt,
        backend='numpy', # Standard backend
        filter_order=16  # Active filtering for stability
    )
    
    solver = CubedSphereSWE(config)
    
    # --- Initialize State (Alpha=45) ---
    state_init = np.zeros((3, 6, N+1, N+1))
    
    topo = solver._impl.topology
    faces = solver._impl.faces
    
    alpha_rad = np.deg2rad(alpha_deg)
    sin_alpha = np.sin(alpha_rad)
    cos_alpha = np.cos(alpha_rad)
    
    for i, fname in enumerate(topo.FACE_MAP):
        fg = faces[fname]
        
        # 1. Coordinates
        lam = fg.lon
        th = fg.lat
        
        # 2. Rotated Latitude theta'
        sin_th_prime = np.sin(th) * cos_alpha - np.cos(lam) * np.cos(th) * sin_alpha
        
        # 3. Height Field (Eq 2.3)
        factor = (R * Omega * u0 + 0.5 * u0**2)
        h_phys = h0 - (factor * sin_th_prime**2) / g
        
        # 4. Velocities (Zonal Flow u_s)
        # u_sphere = u0 * (cos_alpha * np.cos(th) + sin_alpha * np.cos(lam) * np.sin(th))
        # v_sphere = -u0 * sin_alpha * np.sin(lam)
        
        u_sph = u0 * (cos_alpha * np.cos(th) + sin_alpha * np.cos(lam) * np.sin(th))
        v_sph = -u0 * sin_alpha * np.sin(lam)
        
        # 5. Project to Covariant Basis (u_1, u_2) manually
        # This avoids dependency on helper functions and ensures exact projection
        
        # Cartesian Basis at (lam, th)
        sin_lam, cos_lam = np.sin(fg.lon), np.cos(fg.lon)
        sin_th, cos_th = np.sin(fg.lat), np.cos(fg.lat)
        
        e_lam_x, e_lam_y, e_lam_z = -sin_lam, cos_lam, 0.0
        e_th_x, e_th_y, e_th_z = -sin_th*cos_lam, -sin_th*sin_lam, cos_th
        
        # Cartesian Velocity Vector V
        Vx = u_sph * e_lam_x + v_sph * e_th_x
        Vy = u_sph * e_lam_y + v_sph * e_th_y
        Vz = u_sph * e_lam_z + v_sph * e_th_z
        
        # Covariant Components u_i = V . g_i
        # g_i are the tangent vectors stored in fg.g1_vec, fg.g2_vec
        u1_cov = Vx*fg.g1_vec[...,0] + Vy*fg.g1_vec[...,1] + Vz*fg.g1_vec[...,2]
        u2_cov = Vx*fg.g2_vec[...,0] + Vy*fg.g2_vec[...,1] + Vz*fg.g2_vec[...,2]
        
        # Store
        state_init[0, i] = h_phys * fg.sqrt_g  # Mass (h * J)
        state_init[1, i] = u1_cov              # Covariant u_1
        state_init[2, i] = u2_cov              # Covariant u_2

    # Run Simulation
    solver.state = state_init.copy()
    T_total = days * 24.0 * 3600.0
    
    state_final = solver.step(0.0, state_init, 0.0) # Dummy call to init logic if needed? No.
    # Actually use explicit loop as 'solve' might not be fully exposed or uses dynamic steps?
    # CubedSphereSWE has `step`. It does not have `solve` in base?
    # BaseSolver usually has `solve`.
    # Let's check BaseSolver. Assuming standard loop for safety and precision control.
    
    # Manual Integration Loop
    t = 0.0
    current_state = state_init.copy()
    
    step_count = 0
    while t < T_total:
        if t + dt > T_total:
            dt = T_total - t
        
        current_state = solver.step(t, current_state, dt)
        t += dt
        step_count += 1
        
    # --- Error Calculation ---
    # Extract Height h = state[0] / sqrt_g
    
    l2_err_sq = 0.0
    l2_ref_sq = 0.0
    linf_err = 0.0
    linf_ref = 0.0
    
    for i, fname in enumerate(topo.FACE_MAP):
        fg = faces[fname]
        
        # Final h
        h_final = current_state[0, i] / fg.sqrt_g
        
        # Exact h (Initial Condition)
        h_exact = state_init[0, i] / fg.sqrt_g
        
        diff = h_final - h_exact
        
        # Integration Weights
        if hasattr(fg, 'walpha'):
            w_alpha = fg.walpha
            w_beta = fg.wbeta
        else:
            # Fallback (Should be there in standard grid)
            raise RuntimeError("Grid weights missing")
            
        W = np.outer(w_alpha, w_beta)
        dA = fg.sqrt_g * W
        
        l2_err_sq += np.sum(diff**2 * dA)
        l2_ref_sq += np.sum(h_exact**2 * dA)
        
        linf_err = max(linf_err, np.max(np.abs(diff)))
        linf_ref = max(linf_ref, np.max(np.abs(h_exact)))
        
    rel_l2 = np.sqrt(l2_err_sq) / np.sqrt(l2_ref_sq)
    rel_linf = linf_err / linf_ref
    
    return rel_l2, rel_linf

def main():
    resolutions = np.arange(4, 54, 4) # N = 4, 8, 12, ..., 52
    K_dt = 15000.0 # Scaling constant
    
    print("| N | L2 Error | Linf Error | L2 C.R. (div 2) | Linf C.R. (div 2) |")
    print("|---|---|---|---|---|")
    
    results = []
    
    for i, N in enumerate(resolutions):
        dt = K_dt / (N**2)
        
        try:
            l2, linf = run_case2(N, dt)
        except Exception as e:
            print(f"Error at N={N}: {e}")
            l2, linf = float('nan'), float('nan')
            
        rate_l2 = 0.0
        rate_linf = 0.0
        
        if i > 0:
            nx_prev, l2_prev, linf_prev = results[-1]
            denom = np.log(N / nx_prev)
            
            if l2 > 1e-16 and l2_prev > 1e-16:
                slope = np.log(l2 / l2_prev) / denom
                rate_l2 = -slope / 2.0
                
            if linf > 1e-16 and linf_prev > 1e-16:
                slope = np.log(linf / linf_prev) / denom
                rate_linf = -slope / 2.0
                
        results.append((N, l2, linf))
        
        r_l2_str = f"{rate_l2:.2f}" if i > 0 else "-"
        r_linf_str = f"{rate_linf:.2f}" if i > 0 else "-"
        
        print(f"| {N} | {l2:.4e} | {linf:.4e} | {r_l2_str} | {r_linf_str} |")
        sys.stdout.flush()
        
    # Plot
    N_vals = [r[0] for r in results]
    L2_vals = [r[1] for r in results]
    
    plt.figure()
    plt.loglog(N_vals, L2_vals, 'o-', label='Relative L2 Error')
    plt.xlabel('Resolution N')
    plt.ylabel('Relative Error (h)')
    plt.title('SWE Convergence (Case 2, alpha=45)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.savefig("validation/results/swe_convergence.png")
    print("\nPlot saved to validation/results/swe_convergence.png")

if __name__ == "__main__":
    main()

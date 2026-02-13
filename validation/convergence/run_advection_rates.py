"""
Advection Convergence Test (High-Precision / Gaussian)
------------------------------------------------------
Based on untitle3.ipynb reference. 
Uses Nondimensional Units (R=1.0) and Gaussian Initial Condition to verify Spectral Convergence.

Key Settings:
- Test Case: Gaussian Pulse (C^inf Smooth)
- Metric: Absolute L2 and Linf Error (Unnormalized)
- Time Integration: LSRK5
- Time Step: dt ~ 1/N^2
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from cubed_sphere.solvers import CubedSphereAdvectionSolver as CubedSphereAdvection, AdvectionConfig

def run_simulation_gaussian(N, CFL=1.0):
    """
    Run Advection with Gaussian IC for one full rotation (T=1.0).
    Returns (L2 Error, Linf Error) (Absolute)
    """
    # Parameters matching untitle3.ipynb
    R = 1.0
    u0 = 2 * np.pi  # 1 rotation in t=1.0
    T_final = 1.0   # 1 Rotation
    
    # Time step estimation strictly following notebook logic
    # dt = (CFL / u0) * (2 / N^2)
    dt = (CFL / u0) * (2.0 / (N*N))
    
    # Configure Solver
    config = AdvectionConfig(
        N=N,
        R=R,
        dt=dt,
        u0=u0,
        alpha0=0.0, # Zonal flow (alpha=0) to match simplest reference case
    )
    
    solver = CubedSphereAdvection(config)
    
    # 1. Exact Solution (Initial Condition)
    # Note: Using solver's internal generation to ensure grid consistency
    state_init = solver.get_initial_condition(type="gaussian", h0=1.0, r0=R/3.0)
    solver.state = state_init.copy()
    
    # 2. Run
    state_final = solver.solve((0.0, T_final), state_init)
    
    # 3. Compute Errors (Absolute)
    diff = state_final - state_init
    
    # Linf (Max Abs Diff)
    # diff shape: (n_vars, 6, N+1, N+1)
    linf = np.max(np.abs(diff))
    
    # L2 (Root Mean Square Integral)
    l2_sq = 0.0
    
    for i, fname in enumerate(solver.topology.FACE_MAP):
        fg = solver.faces[fname]
        
        # Integration Weights
        # Notebook: W = np.outer(fg.walpha, fg.wbeta)
        if hasattr(fg, 'walpha'):
            w_alpha = fg.walpha
            w_beta = fg.wbeta
        else:
            # Fallback if not exposed (though usually is)
            # Assuming LGL weights are attached
            raise RuntimeError("Weights not found on FaceGrid")
            
        W = np.outer(w_alpha, w_beta)
        dA = fg.sqrt_g * W
        
        # Sum (diff^2 * dA)
        # diff is (1, S, N, N), we are at face i
        local_diff = diff[0, i] 
        l2_sq += np.sum((local_diff**2) * dA)
        
    l2 = np.sqrt(l2_sq)
    
    return l2, linf

def main():
    # Resolutions from notebook recommendation (Even numbers)
    # Range(4, 26, 4) -> 4, 8, 12, 16, 20, 24
    resolutions = np.arange(4, 54, 4)
    
    print("| N | L2 Error | Linf Error | L2 C.R. (div 2) | Linf C.R. (div 2) |")
    print("|---|---|---|---|---|")
    
    results = []
    
    for i, N in enumerate(resolutions):
        try:
            l2, linf = run_simulation_gaussian(N)
        except Exception as e:
            print(f"Error at N={N}: {e}")
            l2, linf = float('nan'), float('nan')
            
        rate_l2 = 0.0
        rate_linf = 0.0
        
        if i > 0:
            nx_prev, l2_prev, linf_prev = results[-1]
            
            # Rate = - (log(err) - log(prev)) / (log(N) - log(prev))
            # Same as code: np.log(l2 / l2_prev) / np.log(n / n_prev) -> Slope (Negative)
            # Rate = -Slope
            
            denom = np.log(N / nx_prev)
            
            if l2 > 1e-16 and l2_prev > 1e-16:
                slope_l2 = np.log(l2 / l2_prev) / denom
                rate_l2 = -slope_l2 / 2.0
            
            if linf > 1e-16 and linf_prev > 1e-16:
                slope_linf = np.log(linf / linf_prev) / denom
                rate_linf = -slope_linf / 2.0
                
        results.append((N, l2, linf))
        
        # Formatting
        r_l2_str = f"{rate_l2:.2f}" if i > 0 else "-"
        r_linf_str = f"{rate_linf:.2f}" if i > 0 else "-"
        
        print(f"| {N} | {l2:.4e} | {linf:.4e} | {r_l2_str} | {r_linf_str} |")
        sys.stdout.flush()

    # Plot
    N_vals = [r[0] for r in results]
    L2_vals = [r[1] for r in results]
    Linf_vals = [r[2] for r in results]
    
    plt.figure(figsize=(10,10))
    plt.loglog(N_vals, L2_vals, 'o-', label='L2 Error')
    plt.loglog(N_vals, Linf_vals, 's--', label='Linf Error')
    
    # Reference Line O(N^-8)
    if len(N_vals) > 0:
        ref_x = np.array(N_vals)
        # Anchor at first point
        ref_y = L2_vals[0] * (ref_x[0]/ref_x)**8
        plt.loglog(ref_x, ref_y, 'k:', alpha=0.5, label='Reference $O(N^{-8})$')
    
    plt.xlabel('Resolution N')
    plt.ylabel('Absolute Error')
    plt.title('Advection Convergence (Gaussian, R=1.0)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.savefig("validation/results/advection_convergence_gaussian.png")
    print(f"\nPlot saved to validation/results/advection_convergence_gaussian.png")

if __name__ == "__main__":
    main()

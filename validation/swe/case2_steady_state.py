"""
Williamson Case 2: Global Steady State Zonal Flow (Geostrophic Balance)
-----------------------------------------------------------------------
Validation Step 3: Tests the Shallow Water Equations solver for
conservation of geostrophic balance.

Physics:
- Steady state solution to SWE.
- Flow angle alpha allows testing over poles.
- Coriolis force is active.

Ref:
Williamson et al. (1992). "A standard test set for numerical approximations
to the shallow water equations in spherical geometry."
J. Comput. Phys, 102, 211-224.
"""

import numpy as np
import argparse
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from cubed_sphere.solvers import CubedSphereSWE, SWEConfig

def main():
    parser = argparse.ArgumentParser(description="Williamson Case 2: Steady State Zonal Flow")
    parser.add_argument("--N", type=int, default=32, help="Polynomial order")
    parser.add_argument("--alpha", type=float, default=0.0, help="Flow angle in degrees")
    parser.add_argument("--days", type=float, default=5.0, help="Simulation duration in days")
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "jax"])
    parser.add_argument("--dt", type=float, default=None, help="Force time step (optional)")
    args = parser.parse_args()

    # Constants
    R = 6.37122e6
    Omega = 7.292e-5
    g = 9.80616
    alpha_rad = np.deg2rad(args.alpha)
    
    # Physics Parameters
    # u0 = 2 * pi * R / (12 days)
    u0 = 2.0 * np.pi * R / (12.0 * 24.0 * 3600.0) # ~38.6 m/s
    h0 = 2998.0 # Ref height

    print(f"--- Williamson Case 2 (Steady State) ---")
    print(f"Backend: {args.backend}")
    print(f"Resolution N: {args.N}")
    print(f"Alpha: {args.alpha} deg")
    print(f"Duration: {args.days} days")

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
    
    # --- Initialization Logic (Handling Alpha properly) ---
    print("Generating Initial Conditions...")
    state_init = np.zeros((3, 6, args.N+1, args.N+1))
    
    topo = solver._impl.topology
    faces = solver._impl.faces
    
    sin_alpha = np.sin(alpha_rad)
    cos_alpha = np.cos(alpha_rad)
    
    # Precompute initialization
    for i, fname in enumerate(topo.FACE_MAP):
        fg = faces[fname]
        
        # 1. Coordinate Grids
        lam = fg.lon # Lambda (Longitude)
        th = fg.lat  # Theta (Latitude)
        
        # 2. Rotated Latitude theta' (for Height field)
        # sin(theta') = sin(theta)cos(alpha) - cos(theta)cos(lambda)sin(alpha)
        sin_th_prime = np.sin(th) * cos_alpha - np.cos(lam) * np.cos(th) * sin_alpha
        
        # 3. Height Field (Eq 2.3)
        # h = h0 - (1/g)(R Omega u0 + u0^2/2)(sin(theta'))^2
        factor = (R * Omega * u0 + 0.5 * u0**2)
        h_phys = h0 - (factor * sin_th_prime**2) / g
        
        # Store Mass Variable: h * sqrt_g
        state_init[0, i] = h_phys * fg.sqrt_g
        
        # 4. Velocity Field (Eq 2.1, 2.2)
        # u_s (zonal) = u0 (cos(alpha)cos(theta) + sin(alpha)cos(lambda)sin(theta))
        # v_s (merid) = -u0 sin(alpha) sin(lambda)
        
        u_sph = u0 * (cos_alpha * np.cos(th) + sin_alpha * np.cos(lam) * np.sin(th))
        v_sph = -u0 * sin_alpha * np.sin(lam)
        
        # 5. Project to Covariant Components u_1, u_2
        # Cartesian components of V (m/s)
        sin_l, cos_l = np.sin(lam), np.cos(lam)
        sin_t, cos_t = np.sin(th), np.cos(th)
        
        Vx = u_sph * (-sin_l)            + v_sph * (-sin_t * cos_l)
        Vy = u_sph * (cos_l)             + v_sph * (-sin_t * sin_l)
        Vz = u_sph * (0.0)               + v_sph * (cos_t)
        
        # Broadcasting for dot product
        # g1_vec shape: (N+1, N+1, 3)
        b1 = Vx * fg.g1_vec[..., 0] + Vy * fg.g1_vec[..., 1] + Vz * fg.g1_vec[..., 2]
        b2 = Vx * fg.g2_vec[..., 0] + Vy * fg.g2_vec[..., 1] + Vz * fg.g2_vec[..., 2]
        
        state_init[1, i] = b1
        state_init[2, i] = b2

    # --- Run Simulation ---
    print("Starting integration...")
    # solver.state = state_init.copy()
    
    t0 = time.time()
    # state_final, t_final = solver.integrate(args.days * 24.0 * 3600.0)
    T_total = args.days * 24.0 * 3600.0
    state_final = solver.solve((0.0, T_total), state_init) # Returns just state
    t1 = time.time()
    
    print(f"Integration complete in {t1-t0:.2f}s")
    
    # --- Error Analysis ---
    
    error_state = state_final - state_init
    
    # Accumulators
    h_err_L2_sq = 0.0
    h_norm_L2_sq = 0.0
    h_err_L1 = 0.0
    h_err_Linf = 0.0
    
    total_area = 0.0
    
    u_err_L2_sq = 0.0
    u_norm_L2_sq = 0.0
    
    for i, fname in enumerate(topo.FACE_MAP):
        fg = faces[fname]
        sqrt_g = fg.sqrt_g
        w = fg.walpha # Integration weights (GL points)
        
        # Mass Error
        diff_mass = error_state[0, i]
        h_phys_err = diff_mass / sqrt_g
        h_phys_exact = state_init[0, i] / sqrt_g
        
        # Area Element dA for sum = sqrt_g * w_i * w_j ?
        # Wait, typically sum(f * dA) approx sum(f_ij * w_i * w_j * sqrt_g_ij).
        # But 'w' provided by grids usually sums to 2 (Legendre Gauss Lobatto weights on [-1,1]).
        # So yes, we need sqrt_g.
        
        W2D = np.outer(w, w) # (N+1, N+1)
        dA = sqrt_g * W2D
        
        # L2 Norm Height
        h_err_L2_sq += np.sum( (h_phys_err**2) * dA )
        h_norm_L2_sq += np.sum( (h_phys_exact**2) * dA )
        
        # L1 Norm Height
        h_err_L1 += np.sum( np.abs(h_phys_err) * dA )
        
        # Linf Norm Height
        h_err_Linf = max(h_err_Linf, np.max(np.abs(h_phys_err)))
        
        total_area += np.sum(dA)
        
        # Velocity Error
        du1 = error_state[1, i]
        du2 = error_state[2, i]
        
        u1_exact = state_init[1, i]
        u2_exact = state_init[2, i]
        
        # Inverse Metric g^ij
        g11 = fg.g_ij[..., 0, 0]
        g12 = fg.g_ij[..., 0, 1]
        g22 = fg.g_ij[..., 1, 1]
        det = g11*g22 - g12**2
        
        inv_g11 = g22 / det
        inv_g12 = -g12 / det
        inv_g22 = g11 / det
        
        # Squared Magnitude of Difference vector
        # |du|^2 = du_i * du^i
        diff_sq = (du1 * (inv_g11 * du1 + inv_g12 * du2) + 
                   du2 * (inv_g12 * du1 + inv_g22 * du2))
                   
        exact_sq = (u1_exact * (inv_g11 * u1_exact + inv_g12 * u2_exact) + 
                    u2_exact * (inv_g12 * u1_exact + inv_g22 * u2_exact))
        
        u_err_L2_sq += np.sum(diff_sq * dA)
        u_norm_L2_sq += np.sum(exact_sq * dA)

    # Normalize L2
    h_rel_L2 = np.sqrt(h_err_L2_sq) / np.sqrt(h_norm_L2_sq)
    u_rel_L2 = np.sqrt(u_err_L2_sq) / np.sqrt(u_norm_L2_sq)
    h_rmse = np.sqrt(h_err_L2_sq / total_area)
    
    print("\n--- Results ---")
    print(f"Total Area: {total_area:.4e}")
    print(f"Height L2 Rel Error:   {h_rel_L2:.4e}")
    print(f"Height L1 Error:       {h_err_L1:.4e}")
    print(f"Height Linf Error:     {h_err_Linf:.4e} m")
    print(f"Height RMSE:           {h_rmse:.4e} m")
    print(f"Velocity L2 Rel Error: {u_rel_L2:.4e}")
    
    # Assertions for Validation
    # Case 2 with alpha=0 (default) is trivial-ish but tests Coriolis balance.
    # Relative errors should be small.
    
    if h_rel_L2 < 1e-4:
        print("PASS: Height L2 Error < 1e-4")
    else:
        print("WARNING: Height Error larger than expected.")
        
    mass_init = 0.0
    mass_final = 0.0
    
    for i, fname in enumerate(topo.FACE_MAP):
        fg = faces[fname]
        w = fg.walpha
        W2D = np.outer(w, w)
        mass_init += np.sum(state_init[0, i] * W2D)
        mass_final += np.sum(state_final[0, i] * W2D)
        
    mass_err = (mass_final - mass_init) / mass_init
    print(f"Global Mass Relative Error: {mass_err:.4e}")
    
    if abs(mass_err) < 1e-12:
        print("PASS: Mass effectively conserved")
    else:
        print("WARNING: Mass Drift detected.")

if __name__ == "__main__":
    main()

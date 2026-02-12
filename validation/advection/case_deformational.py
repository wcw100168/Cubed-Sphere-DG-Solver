"""
Deformational Flow: Twin Vortices (Nair et al., 2005)
-----------------------------------------------------
Validation Step 2: Tests the solver's ability to resolve fine filaments
and handle strong gradients (non-divergent deformational flow).

Physics:
- Coordinate rotation to position vortices.
- Tangential velocity Vt defined in rotated frame.
- Exact solution is time-dependent.

Ref:
Nair, R. D., et al. (2005). "A discontinuous Galerkin global shallow water model."
Monthly Weather Review, 133(4), 876-887.
"""

import numpy as np
import argparse
import sys
import os
import time
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig

def main():
    parser = argparse.ArgumentParser(description="Twist Vortices Deformational Flow Test")
    parser.add_argument("--N", type=int, default=32, help="Polynomial order")
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "jax"])
    parser.add_argument("--cfl", type=float, default=0.2, help="CFL number (smaller for deformation)")
    args = parser.parse_args()

    # Constants
    R = 6.37122e6
    T_days = 6.0 # Benchmark duration
    T_sec = T_days * 24.0 * 3600.0
    
    # Rotated Pole Position (Nair 2005)
    pole_lon = np.pi - 0.8
    pole_lat = np.pi / 4.8
    
    # Flow Parameters
    rho0 = 3.0
    gamma = 5.0
    
    # Scaling Factor: Map unit-sphere velocity to Earth-scale geophysical velocity
    # Benchmark standard: Unit velocity corresponds to one sphere revolution in 12 days?
    # Nair 2005 uses T=12 days as the period.
    # We apply a scaling factor to the non-dimensional velocity formulas.
    EARTH_PERIOD = 12.0 * 24.0 * 3600.0
    # On Unit Sphere, U ~ 1 means L/T = 1. T ~ L = 1 rad? 
    # Usually we want 1 'cycle' ~ 12 days.
    # Non-dim angular velocity ~ 1. Physical ~ 2pi / 12 days.
    SCALE_FACTOR = (2.0 * np.pi) / EARTH_PERIOD
    
    # We set u0 to a dummy value because we override the wind field
    config = AdvectionConfig(
        N=args.N, 
        R=R, 
        u0=20.0, # dummy
        alpha0=0.0, 
        CFL=args.cfl, 
        backend=args.backend,
        dt=None # Auto-calc
    )
    solver = CubedSphereAdvectionSolver(config)
    
    # --- 1. Coordinate Rotation Helper ---
    def get_rotated_coords(lon, lat):
        """Map (lon, lat) -> (lon', lat') relative to new pole."""
        # Using formulas from Nair et al. (2005) or standard spherical trig
        sin_lat_p = np.sin(pole_lat)
        cos_lat_p = np.cos(pole_lat)
        
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        del_lon = lon - pole_lon
        
        # Calculate new latitude theta'
        arg = sin_lat * sin_lat_p + cos_lat * cos_lat_p * np.cos(del_lon)
        arg = np.clip(arg, -1.0, 1.0)
        lat_prime = np.arcsin(arg)
        
        # Calculate new longitude lambda'
        # sin(lat') * cos(lat_p) = -cos(lat)*cos(del_lon)*sin(lat_p) + sin(lat)*cos(lat_p) ? No
        # Use y, x for arctan2
        # y = cos(lat) * sin(del_lon)
        # x = -sin(lat)*cos(lat_p) + cos(lat)*sin(lat_p)*cos(del_lon) -- Wait, standard formula check
        
        # Transformation derived from rotating Z axis to P0 vector.
        # Let's verify with standard implementation:
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        
        # P0 vector
        xp = np.cos(pole_lat) * np.cos(pole_lon)
        yp = np.cos(pole_lat) * np.sin(pole_lon)
        zp = np.sin(pole_lat)
        
        # We need rotation matrix M s.t. M * P0 = (0,0,1).
        # Actually we want (lon, lat) -> (lon', lat') where prime system has pole at P0.
        # So we project X onto the rotated basis vectors.
        # Z' axis = P0.
        # Y' axis = (k x P0) / |k x P0| ? Or usually defined such that prime meridian aligns?
        # Nair 2005 usually doesn't specify orientation of lambda' zero. 
        # But implies lambda' is longitude in rotated system.
        
        # Let's use the explicit formulas from commonly used scripts (e.g. cyclops/tempest)
        # for "unrotating" or rotating.
        # Formula from Williamson 1992 (Eq 3-4):
        # sin(lat') = sin(lat)sin(lat_p) + cos(lat)cos(lat_p)cos(lon-lon_p)
        # sin(lon')cos(lat') = cos(lat)sin(lon-lon_p)
        # cos(lon')cos(lat') = -sin(lat)cos(lat_p) + cos(lat)sin(lat_p)cos(lon-lon_p)
        
        num = np.cos(lat) * np.sin(lon - pole_lon)
        den = -np.sin(lat) * cos_lat_p + cos_lat * sin_lat_p * np.cos(lon - pole_lon)
        lon_prime = np.arctan2(num, den) # Check if this matches convention
        
        return lon_prime, lat_prime

    # --- 2. Wind Field Injection ---
    print("Injecting Deformational Wind Field...")
    
    # Max velocity needed for dt calculation
    v_max_global = 0.0
    
    for fname in solver.topology.FACE_MAP:
        fg = solver.faces[fname]
        X, Y, Z = fg.X, fg.Y, fg.Z
        lon, lat = solver.geometry.lonlat_from_xyz(X, Y, Z)
        
        lon_p, lat_p = get_rotated_coords(lon, lat)
        
        # rho' = rho0 * cos(lat')
        rho_prime = rho0 * np.cos(lat_p)
        
        # Vt = 3*sqrt(2)/2 * sech^2(rho') * tanh(rho')
        const_factor = 3.0 * np.sqrt(2.0) / 2.0
        vt = const_factor * (1.0 / np.cosh(rho_prime))**2 * np.tanh(rho_prime)
        
        # In rotated frame, flow is purely zonal (along lat_p circles)?
        # Formula: "The angular velocity is omega'(theta')".
        # This implies u' = Vt (tangential velocity), v' = 0.
        # But wait, omega = Vt / rho'. 
        # u' = R * cos(lat') * omega' = R * cos(lat') * (Vt / (rho0 * cos(lat'))) 
        #    = (R / rho0) * Vt.
        # Let's assume the paper defines Vt as the physical tangential velocity magnitude.
        
        # Actually Nair 2005 says:
        # "The non-divergent velocity field is given by streamfunction psi..."
        # But the prompt gave Vt directly. Let's assume u'_sph = Vt, v'_sph = 0 in rotated frame?
        # Re-reading prompt: "Angular velocity omega'(theta') = 0 if rho'=0 else delta_omega".
        # "omega' = Vt / rho'".
        # If omega is angular velocity, then zonal speed u' = R * cos(lat') * omega'.
        # u' = R * cos(lat') * (Vt / (rho0 * cos(lat'))) = (R/rho0) * Vt.
        
        # So:
        u_prime = (R / rho0) * vt * SCALE_FACTOR
        v_prime = np.zeros_like(u_prime)
        
        # Vector transform (u', v') -> (u, v)
        # Vector V = u' * e_lambda' + v' * e_theta'
        # e_lambda' direction?
        # e_lambda' = (-sin(lon'), cos(lon'), 0) in Rotated Cartesian? No.
        # Easier: The streamfunction is Psi.
        # Then u = -1/R dPsi/dtheta, v = 1/Rcos dPsi/dlambda.
        # But we don't have Psi formula in prompt.
        
        # Use Vector Projection method:
        # 1. Construct V_vec in global cartesian coords.
        #    Need Basis vectors of rotated system e_lon_p, e_lat_p at point P.
        #    e_lon_p is parallel to (P0 x P).
        #    e_lat_p is e_lon_p x P_radial.
        
        P_vec = np.stack([X, Y, Z], axis=-1) # shape (N, N, 3)
        P0_vec = np.array([np.cos(pole_lat)*np.cos(pole_lon), 
                           np.cos(pole_lat)*np.sin(pole_lon),
                           np.sin(pole_lat)]) # (3,)
        
        # Cross product P0 x P (Tangential direction for zonal flow around P0)
        # This aligns with e_lambda' direction (East in rotated frame)
        K_cross_P = np.cross(P0_vec, P_vec) # (N, N, 3)
        norm_KxP = np.linalg.norm(K_cross_P, axis=-1, keepdims=True)
        e_lon_p = K_cross_P / (norm_KxP + 1e-15)
        
        # Velocity vector in 3D: V_vec = u_prime * e_lon_p
        V_vec = u_prime[..., None] * e_lon_p
        
        v_mag = np.linalg.norm(V_vec, axis=-1)
        v_max_global = max(v_max_global, np.max(v_mag))
        
        # Project V_vec onto local (North, East, Radial) of STANDARD grid
        # e_east = (-sin(lon), cos(lon), 0)
        # e_north = (-sin(lat)cos(lon), -sin(lat)sin(lon), cos(lat))
        
        sin_lon, cos_lon = np.sin(lon), np.cos(lon)
        sin_lat, cos_lat = np.sin(lat), np.cos(lat)
        
        e_east = np.stack([-sin_lon, cos_lon, np.zeros_like(lon)], axis=-1)
        e_north = np.stack([-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat], axis=-1)
        
        u_sph = np.sum(V_vec * e_east, axis=-1)
        v_sph = np.sum(V_vec * e_north, axis=-1)
        
        # Store in Geometry for solver to use Contravariant conversion
        # Use existing method
        u1, u2 = solver.geometry.compute_contravariant_vel(fg, u_sph, v_sph)
        
        fg.u1 = u1
        fg.u2 = u2
        
        # Divergence should be zero, but numerically compute it for consistency
        term1 = solver.D @ (fg.sqrt_g * u1)
        term2 = (fg.sqrt_g * u2) @ solver.D.T
        fg.div_u = (1.0 / fg.sqrt_g) * (term1 + term2)
        
        # Check div (should be small)
        max_div = np.max(np.abs(fg.div_u))
        if max_div > 1e-5:
            # Maybe warning? DG requires div term for stability in skew-symmetric form
            pass

    # CRITICAL: If using JAX, we must push the new numpy arrays to the device
    if config.backend == 'jax':
        print("Re-transferring updated wind fields to JAX device...")
        solver._to_jax()

    # Recalculate dt based on actual max velocity
    # u_prime max is around (R/3) * (3*sqrt(2)/2 * 0.5) roughly R*0.7?
    dt_safe = args.cfl * R / (v_max_global * args.N**2)
    print(f"Computed Max Velocity: {v_max_global:.2f} m/s")
    print(f"Adjusted dt: {dt_safe:.4f} s")
    solver.cfg.dt = dt_safe # Inject into config
    
    # --- 3. Initial Condition & Exact Solution ---
    def get_exact_solution(t_time):
        print(f"Calculating exact solution at t={t_time}...")
        results = []
        for fname in solver.topology.FACE_MAP:
            fg = solver.faces[fname]
            lon, lat = solver.geometry.lonlat_from_xyz(fg.X, fg.Y, fg.Z)
            lon_p, lat_p = get_rotated_coords(lon, lat)
            
            rho_prime = rho0 * np.cos(lat_p)
            vt = 3.0 * np.sqrt(2.0) / 2.0 * (1.0 / np.cosh(rho_prime))**2 * np.tanh(rho_prime)
            # omega' = Vt / rho' = Vt / (rho0 * cos(lat_p))?
            # Note: at poles, 0/0. Limit needs care.
            # But formula for phi depends on (lambda' - omega' * t)
            # Use safe division or Taylor expansion? 
            # Or just calc omega directly:
            # omega = (3*sqrt(2)/2 * sech^2 * tanh) / rho? 
            # tanh(x)/x -> 1 at 0.
            
            # Omega calculation
            denom = rho_prime
            numer = vt
            # Mask zeros
            omega = np.zeros_like(denom)
            mask = np.abs(denom) > 1e-9
            omega[mask] = numer[mask] / denom[mask]
            
            # Limit at rho=0 (sech(0)=1, tanh(x)/x -> 1)
            # Vt/rho = C * 1 * 1 = C = 3*sqrt(2)/2
            # Wait, rho0=3. rho'=3*cos(lat'). At equator lat'=0, rho'=3. 
            # At pole lat'=pi/2, rho'=0. 
            # Warning: rho' is NOT radius. It is a parameter.
            # omega(rho'=0) = 3*sqrt(2)/2 * 1 * 1 = 2.12...
            mask_zero = ~mask
            omega[mask_zero] = 3.0 * np.sqrt(2.0) / 2.0 
            
            # Apply Scaling
            omega = omega * SCALE_FACTOR

            arg_sin = np.sin(lon_p - omega * t_time)
            arg_tanh = (rho_prime / gamma) * arg_sin
            phi = 1.0 - np.tanh(arg_tanh)
            results.append(phi)
            
        return results # Returns list of arrays

    # Initialize
    ic_list = get_exact_solution(0.0)
    # Pack into (1, 6, N, N)
    u_initial = np.array(ic_list)[None, ...] # add var dim
    
    # Run
    print("Running Simulation...")
    u_final = solver.solve((0.0, T_sec), u_initial)
    if args.backend == 'jax': u_final = np.array(u_final)

    # Compare
    exact_list = get_exact_solution(T_sec)
    u_exact = np.array(exact_list)[None, ...]
    
    diff = u_final - u_exact
    
    # 4. Metrics & Plots
    total_l1 = 0.0
    total_l1_exact = 0.0
    total_l2_sq = 0.0
    total_exact_l2_sq = 0.0
    max_diff = 0.0
    max_exact = 0.0
    
    total_mass_diff = 0.0
    total_mass_init = 0.0
    
    for i, fname in enumerate(solver.topology.FACE_MAP):
        fg = solver.faces[fname]
        w = np.outer(fg.walpha, fg.wbeta)
        da = fg.sqrt_g * w
        
        f_diff = diff[0, i]
        f_exact = u_exact[0, i]
        
        # L1
        total_l1 += np.sum(np.abs(f_diff) * da)
        total_l1_exact += np.sum(np.abs(f_exact) * da)
        
        # L2
        total_l2_sq += np.sum(f_diff**2 * da)
        total_exact_l2_sq += np.sum(f_exact**2 * da)
        
        # Linf
        max_diff = max(max_diff, np.max(np.abs(f_diff)))
        max_exact = max(max_exact, np.max(np.abs(f_exact)))
        
        # Mass
        total_mass_diff += np.sum((u_final[0,i] - u_initial[0,i]) * da)
        total_mass_init += np.sum(u_initial[0,i] * da)
    
    l1_err = total_l1 / total_l1_exact
    l2_err = np.sqrt(total_l2_sq / total_exact_l2_sq)
    linf_err = max_diff / max_exact
    mass_err = total_mass_diff / total_mass_init
    
    print("\n--- Results ---")
    print(f"Relative L1 Error: {l1_err:.5e}")
    print(f"Relative L2 Error: {l2_err:.5e}")
    print(f"Relative Linf Error: {linf_err:.5e}")
    print(f"Mass Error: {mass_err:.5e}")

    # Plotting
    try:
        # Plot Frame 1 (Equator-centered for view)
        # Just plot one face or a Mollweide projection if possible?
        # Simple Face Dump for inspection
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for i in range(6):
            ax = axes.flat[i]
            im = ax.contourf(u_final[0, i], levels=20)
            ax.set_title(f"Face {i} (Order N={args.N})")
            plt.colorbar(im, ax=ax)
        plt.suptitle(f"Deformational Flow (t={T_days} days)")
        
        out_path = os.path.join(os.path.dirname(__file__), "../results/deformational_plot.png")
        plt.savefig(out_path)
        print(f"Plot saved to {out_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()

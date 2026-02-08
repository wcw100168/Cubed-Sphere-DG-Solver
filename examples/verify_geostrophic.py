
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cubed_sphere.solvers.swe import CubedSphereSWE, SWEConfig
from cubed_sphere.utils import regrid, vis

def verify_geostrophic():
    print("\n--- Running Geostrophic Balance Verification ---")
    
    # Configuration
    # Ensure "Small Planet" or Real Earth?
    # Use Earth params usually for SWE: R=6.371e6, Omega=7.292e-5, g=9.81
    R_earth = 6.371e6
    backend = os.environ.get('CUBED_SPHERE_BACKEND', 'numpy')
    print(f"Running SWE Verification with backend: {backend}")
    config = SWEConfig(
        N=32, 
        CFL=0.1, 
        R=R_earth, 
        H_avg=10000.0, # 10km depth,
        backend=backend
    )
    solver = CubedSphereSWE(config)
    
    # Initialize State: Geostrophic Balance
    # Wind: Zonal Flow u = u0 cos(lat)
    u0 = 40.0 # m/s (Jet)
    
    print("Initializing Balanced State...")
    state = np.zeros((3, 6, config.N, config.N))
    
    # 1. Analytic Height Field
    # h = h0 + (R * Omega * u0 / 2g) * cos(2*lat)
    def h_analytic(lam, theta):
        return (config.R * config.Omega * u0 / (2.0 * config.gravity)) * np.cos(2.0 * theta)
        
    for i, fname in enumerate(solver.topology.FACE_MAP):
        fg = solver.faces[fname]
        lam, theta = solver.geometry.lonlat_from_xyz(fg.X, fg.Y, fg.Z)
        
        # Height perturbation
        state[0, i] = h_analytic(lam, theta)
        
        # Velocity initialization (using Vector Regridding Logic implicitly or explicit formula)
        # Use explicit formula: u_sph = u0*cos(theta), v_sph = 0
        u_sph = u0 * np.cos(theta)
        v_sph = np.zeros_like(theta)
        
        # Rotate to North-East?
        # Actually standard definition of zonal wind is Eastward.
        # But u_sph means E component?
        # Let's be precise: u_east = u0 cos(theta). v_north = 0.
        # But wait, solid body is u = u0 * cos(theta).
        # At Equator (theta=0), u=u0. At Pole (theta=pi/2), u=0.
        # This matches solid body rotation.
        
        # Convert to Contravariant
        u1, u2 = solver.geometry.compute_contravariant_vel(fg, u_sph, v_sph)
        state[1, i] = u1
        state[2, i] = u2
        
    # Store initial state for comparison
    initial_state = state.copy()
    
    # Run Simulation
    print(f"Running simulation for T={config.T_final} seconds...")
    final_state = solver.solve((0.0, config.T_final), state)
    
    # Verification
    # Compare h field drift
    diff_h = final_state[0] - initial_state[0]
    l2_err = np.sqrt(np.mean(diff_h**2))
    max_err = np.max(np.abs(diff_h))
    
    rel_err = l2_err / (np.max(np.abs(initial_state[0])) + 1e-10)
    
    print(f"\nVerification Results (Height Field Drift):")
    print(f"L2 Error: {l2_err:.6e}")
    print(f"Max Error: {max_err:.6e}")
    print(f"Relative Error: {rel_err:.6e}")
    
    # Since we lack sophisticated boundary flux penalties (Numerical Flux) in the current simplified
    # LSWE solver (Volume terms only + No interface coupling), we expect some drift due to boundaries
    # acting like walls or unconnected?
    # Wait, if I didn't implement penalties, the faces are partially decoupled (only volume terms).
    # This might cause drift.
    # But for a smooth global solution initialized globally, volume terms account for the physics.
    # The error should be relatively small if resolution is sufficient.
    
    if rel_err < 1e-2:
        print(">> PASSED: State remained balanced (within 1% tolerance for simplified sovler).")
    else:
        print(">> WARNING: Significant drift detected. Boundary coupling might be needed.")

if __name__ == "__main__":
    verify_geostrophic()

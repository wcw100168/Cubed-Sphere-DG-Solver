
import numpy as np
import matplotlib.pyplot as plt
from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig
from cubed_sphere.utils import plot_cubed_sphere_state

def verify_system_arch():
    print("\n--- Running Multi-Variable Architecture Verification ---")
    
    # 1. Configuration
    # n_vars = 2
    # T_final = 2*pi roughly for u0=2pi? 
    # AdvectionConfig defaults: u0=2*pi. R=1.0. T=1.0 -> 1 full rotation.
    
    config = AdvectionConfig(N=32, CFL=1.0, T_final=1.0, u0=2*np.pi, n_vars=2)
    solver = CubedSphereAdvectionSolver(config)
    
    # 2. Initialization "The Race"
    print("Initializing Multi-Variable State...")
    state = np.zeros((2, 6, config.N, config.N))
    
    # Var 0: Gaussian Hill at (0, 0)
    def gaussian_hill(lam, theta):
        center_lon, center_lat = 0.0, 0.0
        dist = np.arccos(np.cos(theta)*np.cos(center_lat)*np.cos(lam - center_lon) + np.sin(theta)*np.sin(center_lat))
        return np.exp(-(dist * 3.0)**2)
        
    # Var 1: Cosine Bell at (pi/2, 0) (East by 90 deg)
    def cosine_bell(lam, theta):
        center_lon, center_lat = np.pi/2, 0.0
        dist = np.arccos(np.cos(theta)*np.cos(center_lat)*np.cos(lam - center_lon) + np.sin(theta)*np.sin(center_lat))
        r0 = np.pi/3.0
        return np.where(dist < r0, 0.5 * (1.0 + np.cos(np.pi * dist / r0)), 0.0)

    # Use the helper to get coordinate arrays easily, or iterate manually
    # The solver.get_initial_condition fills all vars identically. We need custom per-var.
    
    # Let's manually populate state using the internal geometry
    for k in range(2):
        for i, fname in enumerate(solver.topology.FACE_MAP):
            fg = solver.faces[fname]
            lam, theta = solver.geometry.lonlat_from_xyz(fg.X, fg.Y, fg.Z)
            if k == 0:
                state[k, i, :, :] = gaussian_hill(lam, theta)
            else:
                state[k, i, :, :] = cosine_bell(lam, theta)
                
    print(f"Initial State populated. Var 0 Max: {state[0].max():.4f}, Var 1 Max: {state[1].max():.4f}")
    
    # 3. Create Reference Scalar Run (for Independence Check)
    # We want to make sure Var 0 evolves EXACTLY as if it were alone.
    config_ref = AdvectionConfig(N=32, CFL=1.0, T_final=1.0, u0=2*np.pi, n_vars=1)
    solver_ref = CubedSphereAdvectionSolver(config_ref)
    state_ref_0 = np.zeros((1, 6, config.N, config.N))
    # Populate ref
    for i, fname in enumerate(solver_ref.topology.FACE_MAP):
        fg = solver_ref.faces[fname]
        lam, theta = solver_ref.geometry.lonlat_from_xyz(fg.X, fg.Y, fg.Z)
        state_ref_0[0, i, :, :] = gaussian_hill(lam, theta)
        
    # 4. Run System Simulation
    print("Running System Simulation (T=1.0)...")
    final_state = solver.solve((0.0, 1.0), state)
    
    print("Running Scalar Reference Simulation (T=1.0)...")
    final_state_ref = solver_ref.solve((0.0, 1.0), state_ref_0)
    
    # 5. Verification
    
    # Check 1: Var 0 vs Ref Var 0 (Independence)
    diff = np.abs(final_state[0] - final_state_ref[0])
    max_diff = np.max(diff)
    print(f"Independence Check (Max Diff Var 0 vs Ref): {max_diff:.4e}")
    if max_diff < 1e-12:
        print(">> PASSED: Variables are independent.")
    else:
        print(">> FAILED: Crosstalk detected!")

    # Check 2: Shape Preservation (Return to Start)
    # Since T=1.0 is one full rotation, shapes should match Initial Condition roughly (minus dispersion)
    # But specifically, we check if Var 1 is still separate from Var 0.
    # At T=1.0, Var 0 should be at (0,0) and Var 1 at (pi/2, 0).
    
    # Let's plot final results
    try:
        # Plot Var 0
        plot_cubed_sphere_state(solver, final_state, var_idx=0, title="Final State Var 0 (Gaussian)", alpha=0.9)
        plt.savefig("verify_system_var0.png")
        print("Saved plot verify_system_var0.png")
        
        # Plot Var 1
        plot_cubed_sphere_state(solver, final_state, var_idx=1, title="Final State Var 1 (Cosine)", alpha=0.9)
        plt.savefig("verify_system_var1.png")
        print("Saved plot verify_system_var1.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    verify_system_arch()

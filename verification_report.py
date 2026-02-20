
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import config as jax_config

# 1. Force CPU and Float64 for rigorous verification
os.environ["JAX_PLATFORM_NAME"] = "cpu"
jax_config.update("jax_platform_name", "cpu")
jax_config.update("jax_enable_x64", True)

# Add repo root to path
import sys
sys.path.append(os.getcwd())

from cubed_sphere.solvers.swe_jax import CubedSphereSWEJax
from cubed_sphere.solvers.swe_numpy import CubedSphereSWENumpy

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def report_topology():
    print_header("STEP 1: TOPOLOGY INSPECTION")
    
    # Initialize with small N
    cfg = {'N': 4}
    solver = CubedSphereSWEJax(cfg)
    indices = solver.neighbor_indices
    
    print(f"Neighbor Index Array Shape: {indices.shape}")
    print("Mapping: (Face, Side, [NeighborFace, NeighborSide, ReverseFlag])")
    print("\n[STDOUT] Topology Array Dump (First Face 'P1' / Face 0):")
    
    # Print readable format
    face_idx = 0
    sides = ["West (0)", "East (1)", "South (2)", "North (3)"]
    for s in range(4):
        n_face = indices[face_idx, s, 0]
        n_side = indices[face_idx, s, 1]
        rev = indices[face_idx, s, 2]
        print(f"  Side {sides[s]}: Connects to Face {n_face}, Side {n_side}, Reverse={rev}")
        
    print("\n[STDOUT] Raw Array (Face 0):")
    print(indices[0])

def report_physics(N=32):
    print_header(f"STEP 2: PHYSICS CORRECTNESS (N={N}, Float64)")
    
    cfg = {
        'N': N,
        'dt': 100.0, # Small dt
        'initial_condition': 'case2' # Steady state zonal flow is good for checking symmetry/errors
    }
    
    print("Initializing NumPy Solver (Reference)...")
    solver_np = CubedSphereSWENumpy(cfg)
    state_np = solver_np.get_initial_condition(type='case2')
    
    print("Initializing JAX Solver (Target)...")
    solver_jax = CubedSphereSWEJax(cfg)
    state_jax = jnp.array(state_np) # Identical IC
    
    # Step both solvers
    steps = 5
    dt = cfg['dt']
    print(f"Running {steps} steps on both solvers...")
    
    t = 0.0
    for _ in range(steps):
        state_np = solver_np.step(t, state_np, dt)
        state_jax = solver_jax.step(t, state_jax, dt) # JIT compiles on first call
        t += dt
        
    # Block and Convert
    state_jax_host = np.array(state_jax)
    
    # Compute Error
    # Convert to Height for meaningful error
    # We need sqrt_g from solver_np faces
    # Assuming faces match order
    
    diff_h_all = []
    
    for f_idx, fname in enumerate(solver_np.topology.FACE_MAP):
        fg = solver_np.faces[fname]
        sqrt_g = fg.sqrt_g
        
        mass_np = state_np[0, f_idx]
        mass_jax = state_jax_host[0, f_idx]
        
        h_np = mass_np / sqrt_g
        h_jax = mass_jax / sqrt_g
        
        diff_h = np.abs(h_np - h_jax)
        diff_h_all.append(np.max(diff_h))
        
    max_diff_h = np.max(diff_h_all)
    
    print("\n[STDOUT] Physics Error Metrics (Height Field):")
    print(f"  Max Height Difference: {max_diff_h:.6e} meters")
    
    if max_diff_h < 1e-9:
        print("✅ SUCCESS: Physics Matches Exactly (< 1nm error).")
    elif max_diff_h < 1e-4:
        print("✅ SUCCESS: Physics Matches Closely (< 0.1mm error).")
    else:
        print("❌ FAILURE: Physics Mismatch detected.")
        # Diagnostics
        print(f"  Max Value (Ref): {np.max(state_np):.4f}")
        print(f"  Max Value (JAX): {np.max(state_jax_host):.4f}")
        
def report_compilation(N=96):
    print_header(f"STEP 3: COMPILATION SPEED (N={N})")
    
    cfg = {'N': N, 'dt': 30.0}
    print(f"Initializing JAX Solver with N={N}...")
    try:
        solver = CubedSphereSWEJax(cfg)
        state = solver.get_initial_condition()
        
        print("Triggering JIT Compilation (running 1 step)...")
        start_time = time.time()
        
        # This will trigger compilation of the `_step_core` and `_compute_rhs_core`
        _ = solver.step(0.0, state, 30.0)
        # Block to ensure compilation is done
        _.block_until_ready()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n[STDOUT] Compilation Time: {duration:.4f} seconds")
        
        if duration < 60.0:
            print("✅ SUCCESS: Assembled and Compiled in < 1 minute.")
        else:
            print("❌ WARNING: Compilation took longer than expected.")
    except Exception as e:
        print(f"❌ COMPILATION FAILED: {str(e)}")

if __name__ == "__main__":
    try:
        report_topology()
        report_physics()
        report_compilation()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

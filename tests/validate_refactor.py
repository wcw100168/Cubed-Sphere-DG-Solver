import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import sys
import os

# Ensure package is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import jax.numpy as jnp
try:
    from cubed_sphere.solvers.swe_numpy import CubedSphereSWENumpy
    from cubed_sphere.solvers.swe_jax import CubedSphereSWEJax
    from cubed_sphere.physics.initialization import get_initial_state
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_initialization():
    config = {
        'N': 4,
        'R': 6371220.0,
        'initial_condition': 2, # Case 2
        'alpha': 0.0,
        'H_avg': 10000.0,
        'gravity': 9.80616,
        'Omega': 7.292e-5
    }

    print("--- Testing NumPy Solver Initialization ---")
    try:
        solver_np = CubedSphereSWENumpy(config)
        state_np = solver_np.get_initial_condition()
        print(f"NumPy State Shape: {state_np.shape}")
        if state_np.shape != (3, 6, 5, 5):
            print("FAIL: Incorrect NumPy shape.")
        else:
            print("PASS: NumPy shape correct.")
            
        # Check values briefly (h should be around 10000 * sqrt_g)
        # Just check not all zeros
        if np.all(state_np == 0):
             print("FAIL: NumPy state is all zeros.")
        else:
             print("PASS: NumPy state has content.")
             
    except Exception as e:
        print(f"FAIL: NumPy Solver Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing JAX Solver Initialization ---")
    try:
        solver_jax = CubedSphereSWEJax(config)
        state_jax = solver_jax.get_initial_condition()
        print(f"JAX State Shape: {state_jax.shape}")
        print(f"JAX State Type: {type(state_jax)}")
        
        if state_jax.shape != (3, 6, 5, 5):
            print("FAIL: Incorrect JAX shape.")
        else:
            print("PASS: JAX shape correct.")

        # Check values
        h_jax = np.array(state_jax[0,0,0,0])
        h_np = state_np[0,0,0,0]
        print(f"Comparing H[0,0,0,0]: NumPy={h_np}, JAX={h_jax}")
        
    except Exception as e:
        print(f"FAIL: JAX Solver Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_initialization()

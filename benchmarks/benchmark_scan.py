
import os
# Force CPU usage for stability in this benchmark, unless one wants to test GPU specifically
# and risk the Metal instability seen earlier.
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cubed_sphere.solvers.swe_jax import CubedSphereSWEJax

def benchmark_lax_scan():
    print("--- Benchmark: JAX Python Loop vs. jax.lax.scan ---")
    try:
        platform = jax.devices()[0].platform
    except:
        platform = "unknown"
    print(f"JAX Platform: {platform}")
    
    # Configuration
    N = 32
    dt = 30.0 # seconds
    num_steps = 200 # Using 200 steps for quick verification in this environment
    # Increase num_steps for more rigorous benchmarking on dedicated hardware
    
    t_start = 0.0
    t_end = t_start + dt * num_steps
    
    config = {
        'N': N,
        'R': 6371220.0,
        'initial_condition': "case2",
        'dt': dt,
        'H_avg': 10000.0,
        'gravity': 9.80616,
        'Omega': 7.292e-5
    }
    
    print(f"Configuration: N={N}, Steps={num_steps}, dt={dt}s")
    
    # Initialize Solver
    solver = CubedSphereSWEJax(config)
    
    # Get Initial State
    initial_state_np = solver.get_initial_condition()
    # Ensure it's JAX array for fairness (though solve handles it)
    initial_state_jax = jax.device_put(initial_state_np)
    
    # --- 1. Correctness Check ---
    print("\n[Correctness Check]")
    
    # Run FAST Path (callbacks=None)
    # Warmup first to exclude compilation time
    print("Running Fast Path (Warmup)...")
    _ = solver.solve((0, dt * num_steps), initial_state_jax, callbacks=None)
    
    print(f"Running Fast Path ({num_steps} steps)...")
    start_time = time.time()
    state_fast = solver.solve((t_start, t_end), initial_state_jax, callbacks=None)
    # Block until ready is implicitly handled by converting to numpy at end of solve usually, 
    # but let's be safe if solve returned jax array (it returns numpy array currently).
    # The solve method in swe_jax.py returns np.array(state) which forces synchronization.
    end_time = time.time()
    time_fast = end_time - start_time
    print(f"Fast Path Time: {time_fast:.4f} s")
    
    # Run SLOW Path (callbacks=[dummy])
    def dummy_callback(t, state):
        pass
        
    print("Running Slow Path (Warmup)...")
    _ = solver.solve((0, dt * num_steps), initial_state_jax, callbacks=[dummy_callback])
    
    print(f"Running Slow Path ({num_steps} steps)...")
    start_time = time.time()
    state_slow = solver.solve((t_start, t_end), initial_state_jax, callbacks=[dummy_callback])
    end_time = time.time()
    time_slow = end_time - start_time
    print(f"Slow Path Time: {time_slow:.4f} s")
    
    # Compare
    # Check max difference
    diff = np.max(np.abs(state_fast - state_slow))
    print(f"Max Difference between Fast and Slow: {diff}")
    
    if diff < 1e-12:
        print("PASS: Fast and Slow paths produce identical results.")
    else:
        print("FAIL: Results diverge!")
        
    # --- 2. Performance Metrics ---
    steps_per_sec_fast = num_steps / time_fast
    steps_per_sec_slow = num_steps / time_slow
    speedup = time_slow / time_fast
    
    print("\n[Performance Metrics]")
    print(f"Fast Path: {steps_per_sec_fast:.2f} steps/sec")
    print(f"Slow Path: {steps_per_sec_slow:.2f} steps/sec")
    print(f"Speedup Factor: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark_lax_scan()

import time
import numpy as np
import sys
import os

# Put package in path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig

def run_benchmark_case(N, backend, T_final=0.05):
    print(f"--- Benchmarking N={N}, Backend={backend} ---")
    try:
        # 1. Setup
        t_setup_start = time.time()
        # T_final dictates the physics simulation end time
        config = AdvectionConfig(
            N=N, R=1.0, u0=2*np.pi, CFL=1.0, 
            T_final=T_final,
            backend=backend
        )
        solver = CubedSphereAdvectionSolver(config)
        state = solver.get_initial_condition()
        
        # Calculate dt used directly by solver to estimate step count
        dt = (config.CFL / config.u0) * (2 / config.N**2)
        # We only count "full" steps for the average performance metric
        estimated_steps = int(T_final / dt)
        if estimated_steps < 1: estimated_steps = 1
        
        t_setup_end = time.time()
        setup_time = t_setup_end - t_setup_start
        
        # 2. Warmup / First Step (Compilation for JAX)
        # Run for a tiny duration (e.g. 2 steps) to trigger JIT
        warmup_T = dt * 1.5 
        t_first_start = time.time()
        solver.solve((0.0, warmup_T), state)
        t_first_end = time.time()
        first_step_time = t_first_end - t_first_start
        
        # 3. Production Run
        print(f"  Target T={T_final} (~{estimated_steps} steps)")
        t_run_start = time.time()
        state = solver.solve((0.0, T_final), state)
        t_run_end = time.time()
        
        total_run_time = t_run_end - t_run_start
        avg_step_time = total_run_time / estimated_steps
        
        print(f"  Setup: {setup_time:.4f}s")
        print(f"  First Step: {first_step_time:.4f}s")
        print(f"  Total Run: {total_run_time:.4f}s")
        print(f"  Avg Step: {avg_step_time:.4f}s")
        
        return {
            "setup": setup_time,
            "first": first_step_time,
            "avg": avg_step_time
        }
        
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    cases = [32, 64]
    backends = ["numpy", "jax"]
    results = {}
    
    print("========================================")
    print("     CUBED SPHERE BENCHMARK SUITE       ")
    print("========================================")
    
    for N in cases:
        for backend in backends:
            key = (N, backend)
            res = run_benchmark_case(N, backend)
            results[key] = res
            print("")
            
    # Generate Summary Table in Console
    print("========================================")
    print("             SUMMARY REPORT             ")
    print("========================================")
    print(f"{'N':<5} {'Backend':<8} {'Setup(s)':<10} {'1st Step(s)':<12} {'Avg Step(s)':<12} {'Speedup (vs np)':<15}")
    print("-" * 75)
    
    for N in cases:
        np_res = results.get((N, 'numpy'))
        jax_res = results.get((N, 'jax'))
        
        # Print NumPy row
        if np_res:
            print(f"{N:<5} {'numpy':<8} {np_res['setup']:<10.4f} {np_res['first']:<12.4f} {np_res['avg']:<12.4f} {'1.0x':<15}")
        else:
             print(f"{N:<5} {'numpy':<8} {'FAILED':<10} {'-':<12} {'-':<12} {'-':<15}")

        # Print JAX row
        if jax_res:
             speedup = np_res['avg'] / jax_res['avg'] if (np_res and jax_res['avg'] > 0) else 0.0
             print(f"{N:<5} {'jax':<8} {jax_res['setup']:<10.4f} {jax_res['first']:<12.4f} {jax_res['avg']:<12.4f} {f'{speedup:.2f}x':<15}")
        else:
             print(f"{N:<5} {'jax':<8} {'N/A':<10} {'(Not installed?)':<25}")

if __name__ == "__main__":
    main()

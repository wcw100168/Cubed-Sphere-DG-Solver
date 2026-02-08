import subprocess
import sys
import os
import time

def run_test(command, description):
    """
    Runs a test command.
    Returns True if successful, False otherwise.
    """
    print(f"Running: {description}...")
    print(f"cmd: {' '.join(command)}")
    
    start_time = time.time()
    
    env = os.environ.copy()
    # Ensure JAX uses CPU and x64 for tests to prevent Metal instability/precision issues
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "True"
    
    try:
        # Capture stdout/stderr but print if failed
        result = subprocess.run(
            command,
            cwd=os.path.join(os.path.dirname(__file__), ".."), # Run from repo root
            env=env,
            capture_output=True,
            text=True
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ PASS ({duration:.2f}s)")
            return True
        else:
            print(f"❌ FAIL ({duration:.2f}s)")
            print("--- STDOUT ---")
            print(result.stdout)
            print("--- STDERR ---")
            print(result.stderr)
            print("----------------")
            return False
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        return False

def main():
    print("========================================")
    print("      CUBED SPHERE VALIDATION SUITE     ")
    print("========================================")
    
    python_exe = sys.executable
    
    tests = [
        # 1. Unit Tests
        {
            "cmd": [python_exe, "-m", "unittest", "tests/test_geometry.py"],
            "desc": "Geometry & Topology Unit Tests"
        },
        {
            "cmd": [python_exe, "-m", "unittest", "tests/test_advection.py"],
            "desc": "Advection & Conservation Unit Tests"
        },
        {
            "cmd": [python_exe, "-m", "unittest", "tests/test_consistency.py"],
            "desc": "Backend Consistency Unit Tests"
        },

        # 2. Integration Tests
        {
            "cmd": [python_exe, "-m", "unittest", "tests/test_swe_integration.py"],
            "desc": "SWE Integration Tests (NumPy Backend)"
        },
        {
            "cmd": [python_exe, "-m", "unittest", "tests/test_swe_integration_jax.py"],
            "desc": "SWE Integration Tests (JAX Backend)"
        },
        
        # 3. Smoke Tests (Example Scripts)
        {
            "cmd": [python_exe, "examples/run_swe_convergence.py", 
                   "--backend", "numpy", "--hours", "0.01", "--min_n", "8", "--max_n", "8"],
            "desc": "End-to-End Smoke Test: SWE Convergence (NumPy)"
        },
        {
            "cmd": [python_exe, "examples/run_swe_convergence.py", 
                   "--backend", "jax", "--hours", "0.01", "--min_n", "8", "--max_n", "8"],
            "desc": "End-to-End Smoke Test: SWE Convergence (JAX)"
        },
        {
            "cmd": [python_exe, "examples/run_advection.py"],
            "desc": "End-to-End Smoke Test: Advection Solver"
        }
    ]
    
    failures = []
    
    for test in tests:
        success = run_test(test["cmd"], test["desc"])
        if not success:
            failures.append(test["desc"])
            
    print("\n========================================")
    if failures:
        print(f"Validation FAILED with {len(failures)} errors:")
        for f in failures:
            print(f"- {f}")
        sys.exit(1)
    else:
        print("All Validation Tests PASSED Successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()

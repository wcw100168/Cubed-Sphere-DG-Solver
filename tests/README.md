# Cubed-Sphere DG Solver - Test Suite

This directory contains the comprehensive test suite for the Cubed-Sphere Discontinuous Galerkin (DG) solver. It verifies numerical kernels, grid geometry, physics solvers (Advection & SWE), and backend consistency (NumPy vs JAX).

## 🚀 Quick Start
To run the full validation suite (Unit Tests + End-to-End Smoke Tests), execute the main runner:

```bash
python tests/validate_all.py
```

If successful, you will see a summary report with **PASS** for all stages.

---

## 📂 Test Files Inventory
| File | Description | Backend Scope |
|------|-------------|---------------|
| **validate_all.py** | **Entry Point**. Orchestrates the execution of all unit tests and example scripts. Used by CI/CD pipelines to ensure package integrity. | Both |
| **test_numerics.py** | Verifies the core numerical methods, specifically the **Legendre-Gauss-Lobatto (LGL)** nodes, weights, and differentiation matrices. | NumPy |
| **test_geometry.py** | Validates the Cubed-Sphere grid topology. Checks **Jacobian integration** (Surface Area ≈ $4\pi R^2$) and **Face Connectivity** (East–West–North–South neighbor links). | NumPy |
| **test_advection.py** | Tests the **Scalar Advection** solver. Verifies global mass conservation and correct transport logic across face boundaries. | Both |
| **test_consistency.py** | **Backend Consistency Check**. Forces JAX to use `float64` and runs a single time step on both backends to ensure bit-wise (or near bit-wise) identical results. | Both |
| **test_swe_integration.py** | **SWE NumPy Test**. Verifies the Shallow Water Equations solver using **Williamson Case 2** (Steady State). Checks for L2 stability and solver liveness (perturbation response). | NumPy |
| **test_swe_integration_jax.py** | **SWE JAX Test**. Verifies the JAX backend implementation of SWE. Enforces `x64` precision for strict conservation checks (< 1e-14 error tolerance). | JAX |


## 🛠️ Running Specific Tests

You can run individual test modules using the standard Python `unittest` command:

### Run Core Numerics & Geometry

```bash
python -m unittest tests/test_numerics.py

python -m unittest tests/test_geometry.py
```

### Run Physics Solvers
```bash
# Advection (Scalar Transport)
python -m unittest tests/test_advection.py
# Shallow Water Equations (NumPy)
python -m unittest tests/test_swe_integration.py
# Shallow Water Equations (JAX)
python -m unittest tests/test_swe_integration_jax.py
```

### Run Consistency Checks
```bash
python -m unittest tests/test_consistency.py
```

---

## ⚠️ Notes on JAX Backend
1. **Installation Required**: Tests starting with `test_swe_integration_jax` or using `backend='jax'` will be automatically **skipped** if the `jax` library is not installed in your environment.

2. **Double Precision (x64)**:
   - Strict conservation tests (Error < 1e-12) require Double Precision.
   - The test suite automatically enables `jax.config.update("jax_enable_x64", True)` for these specific tests.
   - If running on hardware that only supports Float32 (some TPU/Mobile chips), conservation tests may fail with larger errors (~1e-4).

## 🔗 Related Examples
The `validate_all.py` script also executes "Smoke Tests" using scripts from the `examples/` directory to ensure end-to-end functionality:
- `examples/run_advection.py`
- `examples/run_swe_convergence.py` (Short duration run)
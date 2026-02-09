# Cubed-Sphere Solver Examples

This directory contains executable scripts demonstrating various capabilities of the Cubed-Sphere solver, from basic "Hello World" simulations to advanced backend configurations and data processing workflows.

## 🚀 Running the Examples

**Important**: These scripts should be executed from the **project root directory** to ensure Python can resolve the `cubed_sphere` package imports correctly.

**Correct Usage:**
```bash
# From the project root
python examples/script_name.py [arguments]
```

**Incorrect Usage:**
```bash
cd examples
python script_name.py  # ModuleNotFoundError
```

---

## 🌊 Full Simulations (Physics)

These scripts run complete physics simulations, serving as both integration tests and scientific validation tools.

### `run_swe_convergence.py`
Runs the **Shallow Water Equations (SWE)** using **Williamson Case 2** (Global Steady State Zonal Flow). This is the primary benchmark for the solver's accuracy.
- **Key Features**:
    - Performs a convergence test by varying grid resolution ($N=8 \sim 32$).
    - Computes global $L_2$ and $L_\infty$ errors.
    - Verifies spectral convergence rates (~Order $N$ or better).
    - Supports both **NumPy** and **JAX** backends.
- **Usage**:
  ```bash
  # Run convergence test using JAX backend
  python examples/run_swe_convergence.py --backend jax --min_n 8 --max_n 32
  ```

### `run_advection.py`
Standard "Hello World" simulation. Solves the scalar **Advection Equation**, transporting a passive tracer (Gaussian hill) around the sphere.
- **Key Features**:
    - Simple setup for understanding the solver loop.
    - Uses the reference NumPy backend.
    - Includes basic real-time visualization options.
- **Usage**:
  ```bash
  python examples/run_advection.py
  ```

---

## 🛠️ Feature Demos (How-To)

Scripts designed to demonstrate specific workflows like I/O, custom initialization, and visualization.

### `demo_custom_input.py`
Demonstrates how to manually initialize the solver state using custom mathematical functions or raw data arrays, rather than the built-in test cases.
- **Key Features**:
    - **Multi-Solver Support**: Works for both `--solver advection` (scalar input) and `--solver swe` (vector state input).
    - Shows how to construct the state array `(3, 6, N, N)` for SWE or `(6, N, N)` for Advection.
- **Usage**:
  ```bash
  # Initialize an SWE simulation with lazy custom inputs
  python examples/demo_custom_input.py --solver swe
  ```

### `demo_offline_io.py`
Demonstrates the production-ready workflow: save simulation data to disk during runtime and post-process it later.
- **Key Features**:
    - Uses `NetCDFMonitor` to save time-series data to `.nc` files.
    - **Multi-Solver Support**: Supports both `--solver advection` and `--solver swe`.
    - Generates MP4 animations from stored NetCDF data.
- **Usage**:
  ```bash
  # Run SWE simulation, save to disk, and animate
  python examples/demo_offline_io.py --solver swe
  ```

### `demo_plotting.py`
Demonstrates how to turn the complex Cubed-Sphere data structure into human-readable global maps.
- **Key Features**:
    - Can generate **Mollweide** (global flat map) or **Orthographic** (globe view) projections.
    - Explains how to use the `cubed_sphere.utils.vis` module.
- **Usage**:
  ```bash
  python examples/demo_plotting.py
  ```

---

## 🧪 Advanced & Utility Demos

Scripts covering hardware acceleration and data interoperability.

### `demo_jax_acceleration.py`
*Formerly `run_jax.py`*
Explicitly demonstrates using the **JAX backend** for high-performance computing.
- **Key Features**:
    - Enables Just-In-Time (JIT) compilation for massive speedups.
    - Shows generic GPU/TPU execution flow.
    - Demonstrates seamless interoperability: converting JAX arrays back to NumPy for standard plotting tools.
- **Usage**:
  ```bash
  python examples/demo_jax_acceleration.py
  ```

### `demo_vector_regridding.py`
*Formerly `verify_wind_field.py`*
Demonstrates how to import external weather data (Lat-Lon grid) onto the Cubed-Sphere grid.
- **Key Features**:
    - Performs vector rotation (East/North wind $\to$ Cubed-Sphere local coordinates).
    - Handles interpolation from standard Lat-Lon grids to the 6-panel structure.
- **Usage**:
  ```bash
  python examples/demo_vector_regridding.py
  ```

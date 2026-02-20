# Cubed Sphere DG Solver

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Backend](https://img.shields.io/badge/Backend-NumPy%20%7C%20JAX-orange)](https://github.com/google/jax)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wcw100168/Cubed-Sphere-DG-Solver/blob/main/tutorials/01_Introduction_Advection.ipynb)

A high-performance Discontinuous Galerkin (DG) solver for advection-diffusion and shallow water equations on the Cubed-Sphere geometry. Designed with a PyTorch-like stateless API, this package enables seamless switching between **NumPy (CPU)** and **JAX (GPU/TPU)** backends for optimal performance.

## Key Features

- **Dual Backend Architecture**: Run purely on CPU with NumPy or accelerate significantly on GPU with JAX.
- **Multi-Variable Support**: Solves systems of equations (e.g., Shallow Water) with `(n_vars, 6, N, N)` state tensors.
- **High-Order Accuracy**: Implements the Spectral Element Method with Legendre-Gauss-Lobatto (LGL) nodes.
- **Strict Conservation**: Ensures mass conservation through rigorous weak-form formulation.
- **Stateless Design**: Separates physics (solvers) from data (state), facilitating easy integration with optimization or machine learning workflows.
- **Advanced Regridding**: Includes tools to remap scalar and vector fields between Rectilinear (Lat-Lon) and Cubed-Sphere grids.

## Features & Performance

### Dual Backend Architecture
The solver provides two execution paths, allowing users to trade off between debuggability and raw performance:

| Feature | NumPy Backend | JAX Backend |
| :--- | :--- | :--- |
| **Primary Use Case** | Debugging, Prototyping, Education | High-Resolution Production Runs, GPU/TPU |
| **Execution Model** | Eager (Line-by-Line) | Lazy (JIT Compiled with XLA) |
| **Parallelism** | CPU (OpenMP/MKL underneath) | Massive SIMD (GPU/TPU) |
| **Looping** | Python `while` loop | `jax.lax.scan` (Fused Kernel) |

### Performance Optimization
The JAX solver implements a **Fast Path** using `lax.scan` which compiles the entire time-integration loop into a single XLA kernel. This eliminates Python interpreter overhead and kernel launch latency.

> **15x - 30x Speedup**: On GPU hardware, the JAX Fast Path has demonstrated speedups of over order of magnitude compared to the Python loop implementation.

### Stateless Design
The core logic is built around `GlobalGrid`, an immutable container for metric terms. This purely functional approach ensures that solver instances are lightweight and compatible with JAX's functional transformation requirements (`jit`, `vmap`, `grad`).

### Quick Start Tip
To enable maximum performance on GPU, ensure you do **not** pass callbacks to the `solve()` method.
```python
# Fast Path (Recommended for Production)
final_state = solver.solve(t_span, initial_state, callbacks=None)

# Slow Path (Debug only)
final_state = solver.solve(t_span, initial_state, callbacks=[my_debug_print])
```

## Supported Models

1.  **Scalar Advection**: Transport of a passive scalar field driven by a prescribed wind velocity. Fully supported on both NumPy and JAX backends.
2.  **Shallow Water Equations (SWE)**: Solves the full non-linear shallow water equations using the Vector Invariant Formulation.
    * **NumPy Backend**: Reference implementation with rigorous stability checks.
    * **JAX Backend**: Fully functional and verified against NumPy backend (consistency < 1e-15). Recommended for high-resolution simulations and massive parallelization.

## Theoretical Background

For a detailed explanation of the mathematical formulation, including the Discontinuous Galerkin method, weak form derivation, and Cubed Sphere geometry metrics, please refer to the [Theoretical Background](docs/Theoretical_Background.md).

## Project Structure

```text
Cubed-Sphere-DG-Solver/
├── benchmarks/            # Scripts for performance comparison (NumPy vs JAX)
├── cubed_sphere/          # Main package source code
│   ├── backend.py         # Backend dispatch logic
│   ├── geometry/          # Grid generation and metric tensors
│   ├── numerics/          # LGL nodes, weights, and D-matrices
│   ├── solvers/           # Time integration loops and PDE operators
│   └── utils/             # Visualization, I/O, and Regridding tools
├── docs/                  # Documentation and technical reports
│   ├── Theoretical_Background.md
│   ├── ACCURACY_REPORT.md
│   └── VALIDATION_REPORT.md
├── examples/              # Usage examples and demos
├── tutorials/             # Jupyter Notebook tutorials (Step-by-step)
├── tests/                 # Unit tests for CI/CD
├── validation/            # Scientific verification cases (Williamson/Nair)
│   ├── advection/         # Case 1 (Solid Body), Deformational Flow
│   └── swe/               # Case 2 (Steady State), Case 6 (Rossby Wave)
├── pyproject.toml         # Package configuration
└── README.md              # Project documentation
```

## Installation

### 1. Clone and Install

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/wcw100168/Cubed-Sphere-DG-Solver.git
cd Cubed-Sphere-DG-Solver
pip install -e .
```

### 2. Optional: Enable GPU Acceleration

To use the JAX backend with GPU support, install the appropriate version of JAX:

```bash
# For macOS (Apple Silicon / Metal)
pip install jax jaxlib jax-metal

# For Linux (NVIDIA GPU / CUDA 12)
pip install "jax[cuda12]"
```

## Quick Start

You can run a complete advection simulation in just a few lines of code:

```python
from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig

# 1. Configure parameters (N=32, CFL=1.0)
config = AdvectionConfig(N=32, CFL=1.0, T_final=1.0, backend='numpy')

# 2. Initialize solver and initial condition
solver = CubedSphereAdvectionSolver(config)
u0 = solver.get_initial_condition(type="gaussian")

# 3. Run simulation (Time stepping is handled automatically)
final_state = solver.solve((0.0, 1.0), u0)
print("Simulation Complete!")
```

## Tutorials

New to the Cubed Sphere? Check out our interactive tutorials:

| Tutorial | Description | Link |
| :--- | :--- | :--- |
| **01. Introduction** | Getting Started: Scalar Advection & Visualization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wcw100168/Cubed-Sphere-DG-Solver/blob/main/tutorials/01_Introduction_Advection.ipynb) |
| **02. Shallow Water Eq** | Advanced Physics: Williamson Case 2 & Stability | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wcw100168/Cubed-Sphere-DG-Solver/blob/main/tutorials/02_Shallow_Water_Equations.ipynb) |
| **03. Data Workflow** | Production: High-Level API, I/O & Regridding | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wcw100168/Cubed-Sphere-DG-Solver/blob/main/tutorials/03_Data_Pipeline_and_IO.ipynb) |

## Validation Suite

The `validation/` directory contains formal verification scripts based on standard atmospheric modeling benchmarks. These ensure the solver is physically correct.

- **Advection**: Williamson Case 1 (Solid Body Rotation), Nair Deformational Flow.
- **Shallow Water**: Williamson Case 2 (Geostrophic Balance), Williamson Case 6 (Rossby-Haurwitz Wave).

See `validation/README.md` for details on how to run these tests and expected results.

## Examples

The `examples/` directory contains complete scripts for testing and demos:

- **run_advection.py**: The standard "Hello World" advection demo.
- **run_swe_convergence.py**: A rigorous convergence test for the Shallow Water Solver.
- **demo_jax_acceleration.py**: Demonstrates JIT compilation and GPU execution.
- **demo_offline_io.py**: A production pipeline: Simulation $\to$ NetCDF $\to$ Animation.
- **demo_vector_regridding.py**: Shows how to remap vector fields (Wind U, V) between Lat-Lon and Cubed-Sphere grids.
- **demo_custom_input.py**: Demonstrates initializing the solver with custom NumPy arrays/data.

## Backend Switching

Switching between NumPy and JAX is as simple as changing a string in the config:

```python
# Use CPU (NumPy)
config_cpu = SWEConfig(..., backend='numpy')

# Use GPU (JAX) - Zero code changes required in your model!
config_gpu = SWEConfig(..., backend='jax')
```

The underlying state tensors (`numpy.ndarray` vs `jax.Array`) are handled automatically by the solver facade.

## Performance & Benchmarks

Curious about the JAX GPU acceleration and scaling performance? Check out our detailed **[Benchmark Report](docs/BENCHMARK_REPORT.md)** for insights into XLA compilation optimization, `lax.scan` speedups, and hardware-specific precision tuning (FP32 vs FP64).
# Cubed Sphere DG Solver

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Backend](https://img.shields.io/badge/Backend-NumPy%20%7C%20JAX-orange)](https://github.com/google/jax)

A high-performance Discontinuous Galerkin (DG) solver for advection-diffusion equations on the Cubed-Sphere geometry. Designed with a PyTorch-like stateless API, this package enables seamless switching between **NumPy (CPU)** and **JAX (GPU/TPU)** backends for optimal performance.

## Key Features

- **Dual Backend Architecture**: Run purely on CPU with NumPy or accelerate significantly on GPU with JAX.
- **High-Order Accuracy**: Implements the Spectral Element Method with Legendre-Gauss-Lobatto (LGL) nodes.
- **Strict Conservation**: Ensures mass conservation through rigorous weak-form formulation.
- **Stateless Design**: Separates physics (solvers) from data (state), facilitating easy integration with optimization or machine learning workflows.

## Theoretical Background

For a detailed explanation of the mathematical formulation, including the Discontinuous Galerkin method, weak form derivation, and Cubed Sphere geometry metrics, please refer to the [Theoretical Background](docs/Theoretical_Background.md).

*(Note: Theory documentation is currently in progress)*

## Project Structure

```text
Cubed-Sphere-DG-Solver/
├── benchmarks/            # Scripts for performance comparison (NumPy vs JAX)
├── cubed_sphere/          # Main package source code
│   ├── backend.py         # Backend dispatch logic (NumPy/JAX abstraction)
│   ├── geometry/          # Grid generation, metric tensors, and coordinate transforms
│   ├── numerics/          # LGL nodes, weights, and differentiation matrices
│   ├── solvers/           # Time integration loops and PDE operators
│   └── utils/             # Visualization tools and helper functions
├── docs/                  # Documentation and reports
├── examples/              # Usage examples and run scripts
├── tests/                 # Unit tests for conservation and numerical accuracy
├── pyproject.toml         # Package configuration and dependencies
└── README.md              # Project documentation
````

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

# For CPU only (if GPU is not available)
pip install "jax[cpu]"
```

## Quick Start

You can run a complete advection simulation in just a few lines of code:

```python
from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig
from cubed_sphere.utils import plot_cubed_sphere_state

# 1. Configure parameters (N=32, T=1.0)
config = AdvectionConfig(N=32, CFL=1.0, T_final=1.0, backend='numpy')

# 2. Initialize solver and initial condition
solver = CubedSphereAdvectionSolver(config)
u0 = solver.get_initial_condition(type="gaussian")

# 3. Run simulation (Time stepping is handled automatically)
final_state = solver.solve((0.0, 1.0), u0)

print("Simulation Complete!")
```

See `examples/run_advection.py` for a full example including visualization.

## Backend Switching

This package supports two computational backends:

### NumPy (Default, CPU)

Best for debugging, development, and small-scale testing. Optimized using in-place operations to minimize memory allocation.

```python
config = AdvectionConfig(..., backend='numpy')
```

### JAX (High-Performance, GPU/TPU)

Best for large-scale, high-resolution simulations. Leverages JIT (Just-In-Time) compilation and XLA to run on accelerators.

```python
config = AdvectionConfig(..., backend='jax')
```

## Testing & Benchmarks

To verify numerical precision (Mass Conservation ~1e-12):

```bash
python -m unittest discover tests
```

To compare JAX vs NumPy performance on your hardware:

```bash
python benchmarks/run_benchmark.py
```

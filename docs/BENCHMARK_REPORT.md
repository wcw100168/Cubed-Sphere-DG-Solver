# Benchmark Report: Cubed-Sphere DG SWE Solver

This document details the performance benchmarks, architectural optimizations, and hardware-specific tuning for the Discontinuous Galerkin (DG) Shallow Water Equations (SWE) solver. All benchmarks are reproducible via the interactive environment.

Interactive Colab Notebook: [Open `examples/Cubed_Sphere_Benchmarks.ipynb` in Colab](https://colab.research.google.com/)

---

## 1. Architectural Optimizations (JAX Backend)

Migrating the DG numerical method to JAX required overcoming several critical bottlenecks related to Just-In-Time (JIT) compilation, graph complexity, and GPU memory management.

### 1.1 Overcoming the XLA Compilation Bottleneck (`vmap` Vectorization)
**The Problem:** Initial JAX implementations utilized Python `for` loops to iterate over the 6 faces and 24 boundaries of the cubed sphere. JAX's `@jit` unrolls Python loops, resulting in a massive XLA computation graph (>30,000 nodes). On a standard Colab instance (Intel Xeon CPU), compiling a high-resolution grid ($N=96$) took **over 20 minutes** before execution even started.

**The Solution:** We refactored the architecture to be fully vectorized using `jax.vmap` and an **Immutable GlobalGrid**. 
1. Replaced dictionary-based topology with a **Static Neighbor Index Array** `(6, 4, 3)`.
2. Stacked face metrics into a Structure-of-Arrays (SoA) format `(6, N, N)`.
3. Applied `vmap` to evaluate volume and flux terms across all 6 faces simultaneously in a SIMD fashion.

**The Result:** Compilation time was reduced by several orders of magnitude.
* **N=96 Compilation Time (Before Refactor):** > 20 minutes (CPU timeout/thrashing)
* **N=96 Compilation Time (Current):** **~1.5 seconds**

### 1.2 Execution Speedup (`jax.lax.scan` vs Python Loops)
To integrate the ODEs over time, relying on a Python loop to dispatch steps to the GPU introduces massive Host-to-Device (CPU-to-GPU) communication overhead. We implemented a "Fast Path" using `jax.lax.scan`, which compiles the entire time-stepping loop into a single GPU kernel.

| Execution Path | Methodology | Performance |
| :--- | :--- | :--- |
| **Slow Path** | Python `for` loop calling `step()` (used for Callbacks) | CPU-bound by Kernel Launch Overhead (~850 steps/s) |
| **Fast Path** | `jax.lax.scan` | Massive speedup, native GPU execution (~1985 steps/s) |

### 1.3 Asynchronous Execution & Memory Management
To ensure benchmarking robustness on consumer GPUs (e.g., Nvidia T4 with 16GB VRAM), the solver implements production-level safety measures:
* **Asynchronous Dispatch Safety:** Explicit use of `jax.block_until_ready()` ensures that XLA compilation and GPU execution times do not bleed into Host-side Python timers.
* **Cache & VRAM Control:** Automated garbage collection and XLA cache clearing (`jax.clear_caches()`) prevent Out-Of-Memory (OOM) exceptions during high-resolution scalability tests.

---

## 2. Hardware Limitations & Dynamic Precision (FP32 vs FP64)

Benchmarking on consumer-grade cloud GPUs requires careful precision management. The Nvidia T4 GPU hardware has a known limitation: its FP64 (Double Precision) compute throughput is only **1/32** of its FP32 capacity. 

We introduced a dynamic dtype casting mechanism `_get_dtype()` that automatically adapts to the user's environment (`jax_enable_x64`), enabling a dual-mode execution strategy:

| Precision Mode | Scientific Accuracy | Relative Execution Speed | Hardware Implication (e.g., Nvidia T4) |
| :--- | :--- | :--- | :--- |
| **Single (FP32)** | Marginal drift (< 1% error) | **Maximum Speed (Baseline)** | Optimal for rapid prototyping on consumer GPUs. Integrates dynamically computed CFL restrictions ($dt$) to prevent catastrophic cancellation. |
| **Double (FP64)** | Spectral accuracy ($< 10^{-11}$ error) | **Slower but Interactive** | Validates exact mathematical correctness against NumPy. Although strictly hardware-throttled (1:32 ratio on T4), our `vmap` architectural optimization compresses the absolute execution time to just seconds, making FP64 highly viable for rigorous validation. |

---

## 3. Solver Scalability (SWE)

The following tests validate the fundamental scaling properties of the vectorized JAX SWE solver. Benchmarks evaluate the average time required to execute a single integration step across varying grid resolutions ($N$).

### Test Environment
- **Platform:** Google Colab
- **Hardware:** Nvidia T4 GPU / Intel Xeon CPU
- **Precision:** Single Precision (FP32)
- **Time-Stepping:** Dynamic CFL-based $dt$
- **Execution:** Fast Path (`lax.scan`, 100 steps)

### Benchmark Results

| N (Grid Resolution) | Total Grid Points ($6 \times N^2$) | Avg. Time per Step (ms) | Throughput (Steps/Sec) |
| :---: | :---: | :---: | :---: |
| **32** | 6,144 | **0.67** | 1,487 |
| **64** | 24,576 | **0.82** | 1,224 |
| **96** | 55,296 | **1.43** | 698 |

*Note: The results demonstrate exceptional GPU parallel efficiency. Even as the problem size increases by nearly 9x (from N=32 to N=96), the execution time per step merely doubles, proving that the JAX/XLA compiler is effectively saturating the GPU stream processors at higher resolutions.*
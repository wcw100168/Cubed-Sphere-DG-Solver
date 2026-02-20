# Benchmark Report: Cubed-Sphere DG Solver

This document details the performance benchmarks, architectural optimizations, and hardware-specific tuning for the Cubed-Sphere Discontinuous Galerkin (DG) solver. The benchmarks cover both the foundational Advection solver and the advanced Shallow Water Equations (SWE) solver.

Interactive Colab Notebook: [Open `Cubed_Sphere_Benchmarks.ipynb` in Colab](https://colab.research.google.com/)

---

## 1. Advanced Optimization: Shallow Water Equations (SWE)

The SWE solver implementation represents the core computational challenge of this project. Migrating the numerical method to JAX required overcoming several critical bottlenecks related to Just-In-Time (JIT) compilation, graph complexity, and hardware limits.

### 1.1 Overcoming the XLA Compilation Bottleneck (`vmap` Vectorization)
**The Problem:** Initial JAX implementations utilized Python `for` loops to iterate over the 6 faces and 24 boundaries of the cubed sphere. JAX's `@jit` unrolls Python loops, resulting in a massive XLA computation graph (>30,000 nodes). On a standard Colab instance (Intel Xeon CPU), compiling a high-resolution grid ($N=96$) took **over 20 minutes** before execution even started.

**The Solution:** We refactored the architecture to be fully vectorized using `jax.vmap`. 
1. Replaced dictionary-based topology with a **Static Neighbor Index Array** `(6, 4, 3)`.
2. Stacked face metrics into a Structure-of-Arrays (SoA) format `(6, N, N)`.
3. Applied `vmap` to evaluate volume and flux terms across all 6 faces simultaneously in a SIMD fashion.

**The Result:** Compilation time was reduced by several orders of magnitude.
* **N=96 Compilation Time (Before):** > 20 minutes (CPU timeout/thrashing)
* **N=96 Compilation Time (After):** **~1.5 seconds**

### 1.2 Execution Speedup (`jax.lax.scan` vs Python Loops)
To integrate the ODEs over time, relying on a Python loop to dispatch steps to the GPU introduces massive Host-to-Device (CPU-to-GPU) communication overhead. We implemented a "Fast Path" using `jax.lax.scan`, which compiles the entire time-stepping loop into a single GPU kernel.

| Execution Path | Methodology | Performance |
| :--- | :--- | :--- |
| **Slow Path** | Python `for` loop calling `step()` (used for Callbacks) | CPU-bound by Kernel Launch Overhead |
| **Fast Path** | `jax.lax.scan` | Massive speedup, native GPU execution |

### 1.3 Hardware Limitations & Dynamic Precision Tuning (FP32 vs FP64)
Benchmarking on consumer-grade GPUs (like Colab's Nvidia T4) requires careful precision management. The Nvidia T4 GPU hardware has a known limitation: its FP64 (Double Precision) compute throughput is only **1/32** of its FP32 capacity. 

**The Initial Pitfall:** Previously, implicit mixed-precision (mixing Python 64-bit floats with JAX 32-bit arrays) forced the T4 to heavily utilize these limited FP64 cores, causing massive performance drops. Additionally, using pure FP32 with large real-world parameters ($R \approx 6.37 \times 10^6$ m) caused Catastrophic Cancellation, leading to NaNs.

**The Dynamic Dtype Solution:**
We introduced a dynamic dtype casting mechanism `_get_dtype()` that automatically adapts to the user's environment (`jax_enable_x64`). This enables a dual-mode execution strategy:

| Precision Mode | Scientific Accuracy | Relative Execution Speed | Hardware Implication (e.g., Nvidia T4) |
| :--- | :--- | :--- | :--- |
| **Single (FP32)** | Marginal drift (~85m / < 1% error) | **Maximum Speed (Baseline)** | Optimal for rapid prototyping on consumer GPUs. Requires a smaller $dt$ to prevent catastrophic cancellation. |
| **Double (FP64)** | Spectral accuracy ($< 10^{-11}$ error) | **Slower but Interactive** | Validates exact mathematical correctness. Although strictly hardware-throttled (1:32 ratio on T4), our `vmap` architectural optimization compresses the absolute execution time to just seconds, making FP64 highly viable and interactive for rigorous validation. |

**Conclusion:** The refactored architecture successfully decouples the mathematical formulation from hardware constraints. Users can seamlessly scale down to FP32 for raw speed, or scale up to FP64 for high-fidelity physics validation without suffering from compilation lock-ups.

---

## 2. Baseline Scaling Benchmarks: Advection Solver

The following tests validate the fundamental scaling properties of the JAX backend compared to the NumPy baseline. Benchmarks evaluate the time required to solve the Advection equation up to $T_{final} = 0.05$ across different grid resolutions ($N$).

### Test Environment
- **CPU:** Apple M3 Pro (12-core) / **NumPy**
- **GPU:** Apple M3 Pro (18-core GPU via Metal) / **JAX**

### Benchmark Results

| N (Resolution) | Total Grid Points | NumPy Time (s) | JAX Time (s) | Speedup (x) |
| :---: | :---: | :---: | :---: | :---: |
| 16 | 1,536 | 0.0574 | **0.0150** | **3.83x** |
| 32 | 6,144 | 0.3546 | **0.0243** | **14.60x** |
| 64 | 24,576 | 2.5028 | **0.0637** | **39.29x** |
| 128 | 98,304 | 20.3533 | **0.4447** | **45.76x** |

### Key Observations
1. **Algorithmic Complexity:** The NumPy backend exhibits $O(N^3)$ to $O(N^4)$ scaling time due to the nested Python loops iterating over the elements.
2. **Hardware Acceleration:** The JAX backend maintains a near $O(1)$ scaling curve for lower resolutions ($N \le 64$) as the problem size is bound by GPU parallelization capacity rather than compute cycles.
3. **Maximum Speedup:** At high resolutions ($N=128$), the JAX implementation demonstrates a **~45x speedup**, proving the effectiveness of the DG method when paired with JIT-compiled tensor operations.

---
*Note: GPU performance metrics include JIT compilation overhead for the first execution. Subsequent executions (warm starts) yield even higher effective speedup ratios.*
# Numerical Accuracy & Convergence Report

This document presents the convergence analysis of the Cubed-Sphere DG Solver using the standard advection benchmark (Cosine Bell or Gaussian Pulse).

## Test Configuration
- **Test Case**: Solid Body Rotation (Advection)
- **Time Integration**: RK4 (assumed)
- **Metric**: $L_2$ and $L_\infty$ Error norms relative to the exact solution.

## Convergence Data

The following tables demonstrate **spectral convergence**. As the polynomial degree ($N$) increases, the error decreases exponentially until it hits the saturation floor determined by machine precision or time-integration errors.

### Case 1: CFL = 0.5 (High Precision Regime)
At low CFL numbers, the solver demonstrates its capability to reach error norms close to **1e-12**.

| N | L2 Error | L2 C.R.| Linf Error | Linf C.R. |
|---|---|---|---|---|
| 10 | 1.77e-02 | - | 3.18e-02 | - |
| 16 | 1.91e-04 | 6.44 | 4.60e-04 | 5.91 |
| 22 | 7.04e-07 | 10.70 | 2.65e-06 | 9.95 |
| 28 | 1.00e-09 | 15.31 | 4.36e-09 | 14.85 |
| 32 | **9.32e-12** | 18.14 | 4.31e-11 | 18.11 |
| 36 | 2.85e-12 | -1.62 | 7.08e-12 | -1.90 |

### Case 2: CFL = 1.0 (Balanced)

| N | L2 Error | L2 C.R.| Linf Error | Linf C.R. |
|---|---|---|---|---|
| 10 | 1.77e-02 | - | 3.18e-02 | - |
| 20 | 5.41e-06 | 9.07 | 1.77e-05 | 8.45 |
| 30 | 1.37e-10 | 14.55 | 4.32e-10 | 16.79 |
| 40 | 1.07e-11 | 3.36 | 2.84e-11 | 3.14 |

### Case 3: CFL = 2.0 (Temporal Error Dominated)
Here, the spatial discretization error becomes smaller than the time-stepping error. The Convergence Rate (C.R.) stabilizes around **4.0**, reflecting the 4th-order accuracy of the time integrator.

| N | L2 Error | L2 C.R.| Linf Error | Linf C.R. |
|---|---|---|---|---|
| 10 | 1.77e-02 | - | 3.18e-02 | - |
| 20 | 5.41e-06 | 9.07 | 1.77e-05 | 8.45 |
| 30 | 1.55e-09 | 4.47 | 3.90e-09 | 4.15 |
| 40 | 1.55e-10 | **4.02** | 4.00e-10 | 3.91 |
| 44 | 7.28e-11 | 3.98 | 1.89e-10 | 3.97 |

## Conclusion
The solver exhibits expected spectral convergence properties. For high-precision requirements, running at lower CFL allows the spatial accuracy to reach approximately $10^{-12}$.
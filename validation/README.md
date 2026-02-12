# Validation Suite for Cubed-Sphere DG Solver

This directory contains formal verification scripts for the Discontinuous Galerkin (DG) solver on the Cubed-Sphere. The tests are based on standard atmospheric modeling benchmarks, specifically the test set proposed by Williamson et al. (1992) and the transport schemes by Nair et al. (2005).

All tests are implemented using the `cubed_sphere` package and target high-order spectral element configurations (`N=32` typical).

## 1. Advection Tests (Scalar Transport)

These tests verify the accuracy and conservation properties of the `CubedSphereAdvection` solver.

| Test Case | Description | Status | Key Metric ($L_2$) | Mass Error |
| :--- | :--- | :--- | :--- | :--- |
| **Case 1** | Solid-Body Rotation (Cosine Bell) | ✅ Verified | $3.24 \times 10^{-3}$ | $\sim 10^{-16}$ |
| **Deformational** | Twin Vortices (Moving Vortices) | ✅ Verified | $5.82 \times 10^{-6}$ | $\sim 10^{-17}$ |

### Case 1: Solid-Body Rotation
Tests advection of a cosine bell scalar field over the poles (cross-pole flow).
*   **Parameters**: $N=32$, $\alpha=45^\circ$, Duration: 12 Days (1 revolution).
*   **Results**:
    *   $L_2$ Error: $3.24 \times 10^{-3}$
    *   Global Mass Error: $~10^{-16}$ (Machine Precision)
*   **Run Command**:
    ```bash
    python3 validation/advection/case1_solid_body.py --N 32 --alpha 45
    ```

### Case: Deformational Flow (Twin Vortices)
Tests the solver's ability to handle strong deformations (filamentation) and reverse flow.
*   **Parameters**: $N=32$, Duration: 6 Days (time-dependent velocity).
*   **Results**:
    *   $L_2$ Error: $5.82 \times 10^{-6}$
    *   Visual: Sharp gradients and filaments are well-preserved.
    *   Global Mass Error: $~10^{-17}$
*   **Run Command**:
    ```bash
    python3 validation/advection/case_deformational.py --N 32
    ```

## 2. Shallow Water Tests (System Dynamics)

These tests verify the full non-linear Shallow Water Equations (SWE) solver, checking momentum conservation, geostrophic balance, and non-linear stability.

| Test Case | Description | Status | Key Metric | Drift |
| :--- | :--- | :--- | :--- | :--- |
| **Case 2** | Steady State Zonal Flow | ✅ Verified | $h_{err} \approx 3.9e^{-11}$ | N/A |
| **Case 6** | Rossby-Haurwitz Wave | ✅ Verified | Stable Integration | $-1.5e^{-9}$ |

### Case 2: Global Steady State Zonal Flow
Tests the ability to maintain exact geostrophic balance.
*   **Parameters**: $N=32$, $\alpha=0^\circ$, Duration: 5 Days.
*   **Results**:
    *   Height field $L_2$ Error: $3.88 \times 10^{-11}$
    *   Velocity field $L_2$ Error: $1.25 \times 10^{-9}$
    *   Mass Conservation: Exact within machine precision.
*   **Run Command**:
    ```bash
    python3 validation/swe/case2_steady_state.py --N 32
    ```

### Case 6: Rossby-Haurwitz Wave
A Wavenumber-4 planetary wave that tests non-linear stability over long integrations. This is a demanding test for spectral element dispersion errors.
*   **Parameters**: $N=32$, Wavenumber 4, Duration: 14 Days.
*   **Results**:
    *   **Stability**: Passed 14-day integration with no blow-up.
    *   **Structure**: Wave pattern (Amplitude range [8335m, 10518m]) preserved with minimal dissipation.
    *   **Conservation**: Global Mass Drift $\approx -1.5 \times 10^{-9}$ (Excellent for non-aliased high-order DG).
*   **Run Command**:
    ```bash
    python3 validation/swe/case6_rossby_wave.py --N 32 --days 14
    ```

## References

1.  **Williamson, D. L., et al. (1992).** "A standard test set for numerical approximations to the shallow water equations in spherical geometry." *Journal of Computational Physics*, 102(1), 211-224.
2.  **Nair, R. D., et al. (2005).** "A discontinuous Galerkin transport scheme on the cubed sphere." *Monthly Weather Review*, 133(4), 814-828.

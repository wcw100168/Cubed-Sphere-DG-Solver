# Theoretical Background

This document outlines the mathematical formulation and numerical methods used in the **Cubed-Sphere DG Solver**. The solver is designed to solve the Advection-Diffusion and Shallow Water Equations (SWE) on a spherical surface using a high-order Discontinuous Galerkin (DG) Spectral Element Method.

The core methodology follows the framework described in Nair et al. (2005) for transport schemes, adapted for the full non-linear SWE system using the Vector Invariant Formulation.

---

## 1. The Cubed-Sphere Geometry

### 1.1 Why Cubed-Sphere?

Traditional Latitude-Longitude grids suffer from the **pole problem**, where grid lines converge at the poles, creating singularities and severely restricting time steps due to the CFL condition .

The **Cubed-Sphere** approach avoids this by projecting the sphere onto the six faces of an inscribed cube. This results in a quasi-uniform grid structure that is free of singularities.

### 1.2 Coordinate Transformation (Equiangular)

This solver employs the **Equiangular Central Projection** (gnomonic projection), which is known to produce a more uniform grid distribution compared to the equidistant projection .

For each of the six faces, we define local curvilinear coordinates $(\alpha, \beta) \in [-\pi/4, \pi/4]$. The mapping from local coordinates to Cartesian coordinates $(X, Y, Z)$ on the sphere is given by:


$$
X = r \tan \alpha, \quad Y = r \tan \beta, \quad Z = r
$$

where $r = R / \sqrt{1 + \tan^2 \alpha + \tan^2 \beta}$.

### 1.3 Metric Tensor

The geometry is encapsulated in the covariant metric tensor $g_{ij}$. For the equiangular projection, the metric tensor is:


$$
g_{ij} = \frac{R^2}{\rho^4 \cos^2 \alpha \cos^2 \beta}
\begin{pmatrix}
1 + \tan^2 \alpha & -\tan \alpha \tan \beta \\
-\tan \alpha \tan \beta & 1 + \tan^2 \beta
\end{pmatrix}
$$

where $\rho^2 = 1 + \tan^2 \alpha + \tan^2 \beta$. The Jacobian of the transformation, crucial for area integration, is $\sqrt{g} = \det(g_{ij})^{1/2}$.

---

## 2. Governing Equations

The solver supports two physical models defined on the sphere.

### 2.1 Scalar Advection Equation

Describes the transport of a passive scalar $\phi$ by a known velocity field $\mathbf{u}$. In flux form on curvilinear coordinates:


$$
\frac{\partial}{\partial t} (\sqrt{g} \phi) + \nabla \cdot (\sqrt{g} \mathbf{u} \phi) = 0
$$

This equation is conservative, ensuring the total mass of the scalar is preserved globally .

### 2.2 Shallow Water Equations (Vector Invariant Form)

To accurately handle non-linear planetary dynamics, we use the **Vector Invariant Formulation** of the Shallow Water Equations. This form separates the kinetic energy and vorticity terms, which is advantageous for conservation properties.

**Variables:**

- $h$: Fluid depth (or geopotential height).
- $\mathbf{u}$: Horizontal velocity vector.

**Equations:**

1. **Mass Continuity:**


$$
\frac{\partial}{\partial t}(\sqrt{g} h) + \nabla \cdot (\sqrt{g} h \mathbf{u}) = 0
$$

2. **Momentum (Covariant components):**


$$
\frac{\partial \mathbf{u}}{\partial t} + (\zeta + f) \mathbf{k} \times \mathbf{u} + \nabla \mathcal{K} + \nabla (g h) = 0
$$

Where:

   - $\zeta = \frac{1}{\sqrt{g}} (\frac{\partial u_2}{\partial \alpha} - \frac{\partial u_1}{\partial \beta})$ is the relative vorticity.
   - $f = 2\Omega \sin \theta$ is the Coriolis parameter.
   - $\mathcal{K} = \frac{1}{2} \mathbf{u} \cdot \mathbf{u}$ is the kinetic energy.

---

## 3. Numerical Discretization

### 3.1 Discontinuous Galerkin (DG) Method

We employ a nodal DG method (often called the Spectral Element Method within elements). The domain is partitioned into non-overlapping elements. Within each element, the solution is approximated by high-order Lagrange polynomials based on **Legendre-Gauss-Lobatto (LGL)** nodes .

**Key features:**

- **Local High Order:** High accuracy ($O(N^k)$) within elements.
- **Discontinuous:** Solutions are allowed to be discontinuous across element boundaries.
- **Mass Matrix:** Using LGL quadrature results in a diagonal mass matrix, making explicit time-stepping highly efficient.

### 3.2 Weak Formulation

The equations are multiplied by a test function $\psi$ and integrated over each element $\Omega_e$. Integrating by parts introduces boundary flux terms :


$$
\int_{\Omega_e} \frac{\partial U}{\partial t} \psi \, d\Omega - \int_{\Omega_e} \mathbf{F}(U) \cdot \nabla \psi \, d\Omega + \oint_{\partial \Omega_e} \mathbf{F}^*(U^-, U^+) \psi \, d\Gamma = 0
$$

### 3.3 Numerical Flux (Rusanov)

The term $\mathbf{F}^*(U^-, U^+)$ represents the numerical flux at the interface between elements, coupling the solution. We use the **Rusanov (Lax-Friedrichs) Flux**, which adds a dissipation term proportional to the maximum wave speed to stabilize the scheme :


$$
\mathbf{F}^* = \frac{1}{2} (\mathbf{F}(U^-) + \mathbf{F}(U^+)) - \frac{C}{2} (U^+ - U^-) \mathbf{n}
$$

where $C$ is the maximum local wave speed (e.g., $|\mathbf{u}| + \sqrt{gh}$ for SWE).

---

## 4. Time Integration & Stabilization

### 4.1 Explicit Runge-Kutta
The semi-discrete form is integrated in time using an explicit Runge-Kutta method.

* **LSRK5 (Implemented)**: We employ the Low-Storage 5-stage 4th-order Runge-Kutta scheme (Carpenter and Kennedy, 1994). This scheme is chosen for its large stability region and low memory footprint, which is superior to the 3rd-order scheme used in the original Nair et al. (2005) paper.

### 4.2 Filtering (Boyd-Vandeven)

To control aliasing errors arising from non-linear terms (spectral blocking) and to suppress Gibbs oscillations near sharp gradients, we apply an explicit **Boyd-Vandeven Filter** at the end of each time step. This filter smoothly dampens the highest-frequency modes without affecting the physical solution accuracy .

---

## 5. Verification Benchmarks

The solver is validated against the standard test suite proposed by Williamson et al. (1992):

1. **Case 1 (Advection):** Solid-body rotation of a cosine bell. Verification of mass conservation and transport accuracy .
2. **Case 2 (Steady State):** Zonal geostrophic flow. Verification of the balance between Coriolis and pressure gradient terms.
3. **Case 6 (Rossby-Haurwitz Wave):** A wavenumber-4 planetary wave. Verification of non-linear stability and long-term integration capability.

---

### References

1. **Williamson, D. L., et al. (1992).** "A standard test set for numerical approximations to the shallow water equations in spherical geometry." *Journal of Computational Physics*, 102(1), 211-224.
2. **Nair, R. D., et al. (2005).** "A discontinuous Galerkin transport scheme on the cubed sphere." *Monthly Weather Review*, 133(4), 814-828.
3. **Sadourny, R. (1972).** "Conservative finite-difference approximations of the primitive equations on quasi-uniform spherical grids." *Monthly Weather Review*, 100, 136-144.

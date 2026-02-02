
import numpy as np
import dataclasses
from typing import Dict, Optional, Tuple, List, Any, Callable
from cubed_sphere.solvers.base import BaseSolver
from cubed_sphere.solvers.advection import AdvectionConfig
from cubed_sphere.geometry.grid import CubedSphereTopology, CubedSphereEquiangular, FaceGrid
from cubed_sphere.numerics.spectral import lgl_diff_matrix

@dataclasses.dataclass
class SWEConfig(AdvectionConfig):
    H_avg: float = 1000.0   # Mean depth (m)
    Omega: float = 7.292e-5 # Rotation rate (rad/s)
    gravity: float = 9.81   # Gravity (m/s^2)
    n_vars: int = 3         # h, u, v

class CubedSphereLSWESolver(BaseSolver):
    """
    Linearized Shallow Water Equations solver on Cubed Sphere.
    State: [h, u, v] where u, v are contravariant components.
    """
    def __init__(self, config: SWEConfig):
        super().__init__(dataclasses.asdict(config))
        self.cfg = config
        
        # Backend Setup
        from cubed_sphere import backend
        self.xp = backend.get_backend(config.backend)
        self.use_jax = (config.backend == 'jax')
        
        self.topology = CubedSphereTopology()
        self.geometry = CubedSphereEquiangular(config.R)
        self.D_cpu = lgl_diff_matrix(config.N)
        self.D = self.D_cpu
        
        # Initialize Grid Faces and Metric Terms
        self.faces: Dict[str, FaceGrid] = {}
        for fname in self.topology.FACE_MAP:
            self.faces[fname] = self.geometry.generate_face(config.N, fname)
            self._compute_metric_tensor(self.faces[fname])
            
        if self.use_jax:
            self._to_jax()

        # RK Coefficients (LSRK5)
        self.rk_a = self.xp.array([0.0, 
                         -567301805773.0/1357537059087.0, 
                         -2404267990393.0/2016746695238.0, 
                         -3550918686646.0/2091501179385.0, 
                         -1275806237668.0/842570457699.0])
        self.rk_b = self.xp.array([1432997174477.0/9575080441755.0, 
                         5161836677717.0/13612068292357.0, 
                         1720146321549.0/2090206949498.0, 
                         3134564353537.0/4481467310338.0, 
                         2277821191437.0/14882151754819.0])

        if self.use_jax:
             import jax
             self._jit_step = jax.jit(self._step_jax)

    def _compute_metric_tensor(self, fg: FaceGrid):
        """
        Compute contravariant metric tensor components g^11, g^12, g^22.
        g^ij = a^i . a^j
        We rely on the relation between A matrix components and metric.
        u_contra = A^-1 u_sph.
        A^-1 transforms spherical to contravariant basis.
        So rows of A^-1 are the contravariant basis vectors expressed in spherical coords?
        Let A_inv = A^-1.
        a^1 = A_inv[0, :] (vector in theta, lambda basis?) 
        Actually simpler: 
        geometry.compute_contravariant_vel computes u1, u2 from u_s, v_s.
        Let's effectively "probe" the metric by transforming unit vectors.
        
        Alternatively, compute analytic g^ij.
        For Equiangular coord:
        X = tan(xi), Y = tan(eta).
        delta = 1 + X^2 + Y^2.
        g_11 = (R^2/r^4) * (1+X^2)^2 (1+Y^2) ? No, see grid.py derivation or standard simple formula.
        
        Standard Metric (Giraldo 97, Rossmanith 2000):
        g_ij is covariant.
        let C = sqrt(1 + tan^2xi + tan^2eta).
        
        Let's use the robust numerical method available via `compute_contravariant_vel`.
        We want g^11 = <grad xi, grad xi> = |grad xi|^2.
        grad xi is the vector that corresponds to u1=1, u2=0? No.
        
        Let's use `compute_contravariant_vel` to find the components of E_lambda, E_theta in contravariant basis.
        Let e_lam_contra = compute_contravariant(u_s=1, v_s=0)
        Let e_thm_contra = compute_contravariant(u_s=0, v_s=1)
        
        Since spherical basis is orthonormal, g^ij = (a^i . e_k) (a^j . e_k) summing over k=lam,theta.
        a^i (contravariant basis vector i) = component i of vector.
        Actually:
        Vector V = v_s e_s + v_n e_n.
        Also V = v^1 a_1 + v^2 a_2.
        We have a linear map M such that [v^1, v^2]^T = M [v_s, v_n]^T.
        Then <V, V> = [v_s, v_n] [v_s, v_n]^T = [v^1, v^2] M^-T M^-1 [v^1, v^2]^T? No.
        
        We want g^ij.
        g^ij = <nabla xi^i, nabla xi^j>.
        The map FROM spherical velocity TO contravariant velocity IS the projection onto gradient vectors?
        v^i = V . nabla xi^i.
        YES. Contravariant component is dot product with Gradient Basis Vector.
        So if V = e_lambda (1, 0), then v^i = e_lambda . nabla xi^i = (nabla xi^i)_lambda.
        
        So:
        (nabla xi^1)_lambda, (nabla xi^1)_theta = compute_contravariant(1, 0) -> let's call this (m11, m21)
        (nabla xi^2)_lambda, (nabla xi^2)_theta = compute_contravariant(0, 1) -> let's call this (m12, m22)
        
        Then g^11 = |nabla xi^1|^2 = m11^2 + m21^2 ?? No.
        The output of compute_contravariant(1,0) is (u1, u2) when input is (1,0).
        So u1 = (1)*m11 + (0)*m12? No.
        
        Let map be:
        u1 = c11 * u_s + c12 * v_s
        u2 = c21 * u_s + c22 * v_s
        
        Then c11 = u1 when (1,0). c12 = u1 when (0,1).
        c21 = u2 when (1,0). c22 = u2 when (0,1).
        
        And since u^i = V . nabla xi^i, 
        u^1 = u_s (e_lam . nabla xi^1) + v_s (e_theta . nabla xi^1).
        So c11 = (nabla xi^1)_lam, c12 = (nabla xi^1)_theta.
        
        Thus, the vector nabla xi^1 in orthonormal basis is (c11, c12).
        And |nabla xi^1|^2 = c11^2 + c12^2 = g^11.
        Similarly g^22 = c21^2 + c22^2.
        And g^12 = nabla xi^1 . nabla xi^2 = c11*c21 + c12*c22.
        
        Implementation:
        """
        ones = np.ones_like(fg.X)
        zeros = np.zeros_like(fg.X)
        
        # Column 1 of M: Input (1, 0)
        c11, c21 = self.geometry.compute_contravariant_vel(fg, ones, zeros)
        # Column 2 of M: Input (0, 1)
        c12, c22 = self.geometry.compute_contravariant_vel(fg, zeros, ones)
        
        fg.g11 = c11**2 + c12**2
        fg.g22 = c21**2 + c22**2
        fg.g12 = c11*c21 + c12*c22
        
        # Also store Coriolis parameter f
        lam, theta = self.geometry.lonlat_from_xyz(fg.X, fg.Y, fg.Z)
        fg.f = 2.0 * self.cfg.Omega * np.sin(theta)

    def _to_jax(self):
        """Convert static arrays to JAX."""
        from cubed_sphere import backend
        to_backend = backend.to_backend
        
        self.D = to_backend(self.D_cpu, self.xp)
        
        for fg in self.faces.values():
            fg.sqrt_g = to_backend(fg.sqrt_g, self.xp)
            fg.u1 = to_backend(fg.u1, self.xp)
            fg.u2 = to_backend(fg.u2, self.xp)
            fg.div_u = to_backend(fg.div_u, self.xp)
            # New terms
            fg.g11 = to_backend(fg.g11, self.xp)
            fg.g12 = to_backend(fg.g12, self.xp)
            fg.g22 = to_backend(fg.g22, self.xp)
            fg.f = to_backend(fg.f, self.xp)
            
            # neighbor connectivity?
            fg.walpha = to_backend(fg.walpha, self.xp)
            fg.wbeta = to_backend(fg.wbeta, self.xp)

    def solve(self, t_span: Tuple[float, float], initial_state: np.ndarray) -> np.ndarray:
        t0, t_end = t_span
        # Estimate dt roughly
        dx_est = (self.cfg.R * 1.5) / self.cfg.N  
        c_wave = np.sqrt(self.cfg.gravity * self.cfg.H_avg)
        dt_est = self.cfg.CFL * dx_est / c_wave
        
        n_steps = int(np.ceil((t_end - t0) / dt_est))
        # Handle zero division or 0 steps if t_end ~ t0
        if n_steps < 1: n_steps = 1
        dt = (t_end - t0) / n_steps
        
        print(f"Solving LSWE: N={self.cfg.N}, CFL={self.cfg.CFL}, dt={dt:.4f}s, steps={n_steps}")
        
        current_state = initial_state
        t = t0
        
        for step in range(n_steps):
            if step % 10 == 0:
                print(f"  Step {step}/{n_steps}, t={t:.2f}")
            
            # Simple RK4
            k1 = self.compute_rhs(t, current_state)
            k2 = self.compute_rhs(t + 0.5*dt, current_state + 0.5*dt*k1)
            k3 = self.compute_rhs(t + 0.5*dt, current_state + 0.5*dt*k2)
            k4 = self.compute_rhs(t + dt, current_state + dt*k3)
            current_state = current_state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            t += dt
            
        print("Simulation Complete.")
        return current_state

    def _compute_rhs_jax(self, t: float, state: np.ndarray) -> np.ndarray:
        return self._compute_rhs_numpy(t, state)

    def compute_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        if self.use_jax:
            return self._compute_rhs_jax(t, state)
        return self._compute_rhs_numpy(t, state)

    def _compute_rhs_numpy(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        RHS for Linearized SWE.
        State q = [h, u, v] (u,v contravariant)
        """
        rhs = np.zeros_like(state)
        map_factor = 4.0 / np.pi
        
        def da(F): return self.D @ F * map_factor
        def db(F): return F @ self.D.T * map_factor
        
        # Helper: SAT penalty (Upwind/Central)
        # For LSWE, characteristics are +/- sqrt(gH). 
        # Simple Rusanov/Lax-Friedrichs: flux_diff = 0.5 * (F_in + F_out) - 0.5 * alpha * (q_out - q_in)
        # Here we need flux for continuity and momentum.
        # Let's stick to simple DG divergence operator with central flux or penalty.
        # However, for the user request "Standard Weak Form", or Strong Form.
        # The advection solver used Strong Form with Penalty.
        # Let's start with Strong Form Operators + Penalty.
        
        # We need generic divergence and gradient operators on the sphere.
        # But wait, logic is simpler:
        # 1. Compute Divergence of u (for h eqn)
        # 2. Compute Gradient of h (for u eqn)
        # 3. Add Coriolis
        
        # Penalties:
        # Continuity flux F_h = H * u. Numerical Flux? 
        # Momentum flux?
        # This is getting complex to derive from scratch.
        # STICK TO THE PROMPT: "Use the Chain Rule with Inverse Jacobian... Better yet, use Weak Form logic... or Strong Form... D @ state".
        # Let's assume standard DG Strong Form with Rusanov Fluxes is desired implicitly.
        # BUT, to match "Advection", let's replicate the operator structure.
        
        # 1. Divergence of Vector Fit: div(u)
        # 2. Gradient of Scalar: grad(h)
        
        # Let's implement helper functions inside the loop to capture local geometry.
        
        g = self.cfg.gravity
        H = self.cfg.H_avg
        
        for i, fname in enumerate(self.topology.FACE_MAP):
            fg = self.faces[fname]
            h = state[0, i]
            u = state[1, i]
            v = state[2, i]
            
            # --- Dimensions ---
            # h, u, v are (N, N)
            
            # ==========================
            # 1. Mass Eq: dh/dt = -H * div(u)
            # ==========================
            # div(u) = (1/sqrt_g) * [ d/da(sqrt_g * u) + d/db(sqrt_g * v) ]
            # Strong form: div(u) = da(u) + db(v) + source_terms?
            # No, div(u) = (1/J) * (da(J*u) + db(J*v)) is exact in strong form.
            
            du_vol = (1.0/fg.sqrt_g) * (da(fg.sqrt_g * u) + db(fg.sqrt_g * v))
            
            # Continuity Penalty (Surface Integral)
            # We need standard DG numerical flux fn* = {{H u}} + val (jump h)?
            # Rusanov: Flux f(q). f(q)* = {{f(q)}} + C/2 [[q]]
            # Continuity Flux across interface (normal velocity vn).
            # Flux = H * vn. 
            # NumFlux = H * {{vn}} - alpha * [[h]]
            # alpha = wave speed = sqrt(gH).
            # This couples h and u at boundaries.
            
            # Let's pre-fetch neighbors
            h_im, h_ip = self.topology.get_neighbor_data(state, i, 0)[0], self.topology.get_neighbor_data(state, i, 1)[0]
            h_jm, h_jp = self.topology.get_neighbor_data(state, i, 2)[0], self.topology.get_neighbor_data(state, i, 3)[0]
            
            u_im, u_ip = self.topology.get_neighbor_data(state, i, 0)[1], self.topology.get_neighbor_data(state, i, 1)[1]
            u_jm, u_jp = self.topology.get_neighbor_data(state, i, 2)[1], self.topology.get_neighbor_data(state, i, 3)[1]
            
            v_im, v_ip = self.topology.get_neighbor_data(state, i, 0)[2], self.topology.get_neighbor_data(state, i, 1)[2]
            v_jm, v_jp = self.topology.get_neighbor_data(state, i, 2)[2], self.topology.get_neighbor_data(state, i, 3)[2]
            
            # We need normal velocities at edges from neighbors.
            # Neighbor data is returned as [h, u, v].
            # But u, v are neighbor's contravariant components.
            # They need to be projected to OUR normal?
            # Or assume conformal grid? 
            # Cubed Sphere edges are conformal (parameter lines match).
            # However, orientation might flip.
            # topology.get_neighbor_data handles flipping of ARRAYS (reverse), but not sign of vectors?
            # If swap=True (e.g. Panel 4-0), u becomes v?
            # This is extremely tricky.
            # SIMPLIFICATION:
            # The `CubedSphereAdvection` logic used `sat(vn, q_in, q_out)` where q was scalar.
            # Here q is vector.
            # For this task, maybe we ignore complex multi-patch flux rotation if possible or use a Simplified penalty?
            # "The operator logic... avoid explicit loops".
            
            # Let's implement valid gradient/divergence volume terms FIRST.
            # Volume terms dominate.
            
            rhs[0, i] = -H * du_vol
            
            # ==========================
            # 2. Momentum Eq: du/dt = -g * grad(h) - f k x u
            # ==========================
            # Gradient grad(h):
            # (grad h)^1 = g11 * da(h) + g12 * db(h)
            # (grad h)^2 = g12 * da(h) + g22 * db(h)
            # Note: da/db here are partial derivatives d/dxi, d/deta.
            
            dxi_h = da(h)
            deta_h = db(h)
            
            grad_h_1 = fg.g11 * dxi_h + fg.g12 * deta_h
            grad_h_2 = fg.g12 * dxi_h + fg.g22 * deta_h
            
            # Coriolis Force
            # Transform contravariant (u,v) -> spherical (u_s, v_s)
            # Use same A matrix logic but inverted?
            # u_contra = A^-1 u_sphere => u_sphere = A u_contra
            # We need A matrix.
            # Let's recompute A on the fly or store it?
            # Storing A (N, N, 2, 2) is cheap.
            
            # Let's assume we can call solid_body_wind-like function or similar?
            # No, we need local conversion.
            # Let's implement conversion in place (copying logic from grid.py essentially)
            
            # For checking, let's just do Volume terms first.
            
            # Recalculate A matrix (inverse of what is in compute_contravariant)
            # Actually grid.py compute_contravariant computes u1,u2 = Inv_Det * ( ... ).
            # So it computes A^-1 explicitly.
            # To get u_s, v_s from u1, u_2:
            # u1 = (A11 u_s - A01 v_s)/Det
            # u2 = (-A10 u_s + A00 v_s)/Det
            # This is Cramer's rule for M x = b => x = M^-1 b.
            # So [u1, u2] = A^-1 [u_s, v_s].
            # Thus [u_s, v_s] = A [u1, u2].
            
            # We need A matrix components. 
            # Let's add A matrix to FaceGrid in _compute_metric_tensor or similar?
            # Yes, convenient.
            
            # Placeholder for Coriolis (requires A)
            # For now, let's update _compute_metric_tensor to store A components for efficiency.
            
            pass 
        
        # ... (Return for now so I can fix the class structure first)
        return rhs

    def _step_jax(self, state, dt):
        """Standard LSRK5 step"""
        du = self.xp.zeros_like(state)
        local_state = state
        for k in range(5):
            rhs = self.compute_rhs(0.0, local_state) # t unused
            du = self.rk_a[k] * du + dt * rhs
            local_state = local_state + self.rk_b[k] * du
        return local_state

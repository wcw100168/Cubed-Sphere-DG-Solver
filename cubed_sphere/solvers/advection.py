import numpy as np
import dataclasses
from typing import Dict, Optional, Tuple, List, Any, Callable
from cubed_sphere.solvers.base import BaseSolver
from cubed_sphere.geometry.grid import CubedSphereTopology, CubedSphereEquiangular, FaceGrid
from cubed_sphere.numerics.spectral import lgl_diff_matrix

@dataclasses.dataclass
class AdvectionConfig:
    N: int                   # Polynomial order
    R: float = 1.0           # Sphere radius
    u0: float = 2 * np.pi    # Wind speed
    alpha0: float = 0.0      # Wind angle
    CFL: float = 1.0         # CFL number
    T_final: float = 1.0     # Final time
    backend: str = 'numpy'   # 'numpy' or 'jax'
    n_vars: int = 1          # Number of variables/species

class CubedSphereAdvectionSolver(BaseSolver):
    """
    Discontinuous Galerkin solver for the advection equation on the Cubed Sphere.
    Supports both NumPy (CPU) and JAX (GPU/TPU) backends.
    """
    def __init__(self, config: AdvectionConfig):
        super().__init__(dataclasses.asdict(config))
        self.cfg = config
        
        # Backend Setup
        from cubed_sphere import backend
        self.xp = backend.get_backend(config.backend)
        self.use_jax = (config.backend == 'jax')
        
        self.topology = CubedSphereTopology()
        self.geometry = CubedSphereEquiangular(config.R)
        self.D_cpu = lgl_diff_matrix(config.N) # Keep CPU copy for init
        self.D = self.D_cpu # Initialize D for precomputation (NumPy)
        
        # Initialize Grid Faces
        self.faces: Dict[str, FaceGrid] = {}
        for fname in self.topology.FACE_MAP:
            self.faces[fname] = self.geometry.generate_face(config.N, fname)
            
        # Precompute static wind field
        self._precompute_static_fields(config.alpha0, config.u0)
        
        # Backend Array Conversion
        if self.use_jax:
            self._to_jax()

        # LSRK5 Coefficients (Ensure generic array usage)
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

        # For JAX, we compile the step function
        if self.use_jax:
             import jax
             # JIT compile the RK substep or full step logic if possible
             # Because our RK loop is fixed size (5), we can functionalize one full dt step
             self._jit_step = jax.jit(self._step_jax)

    def _to_jax(self):
        """Convert all static arrays to JAX DeviceArrays."""
        from cubed_sphere import backend
        to_backend = backend.to_backend
        
        # Convert Diff Matrix
        self.D = to_backend(self.D_cpu, self.xp)
        
        # Convert FaceGrid fields
        for fg in self.faces.values():
            fg.alpha = to_backend(fg.alpha, self.xp)
            fg.beta = to_backend(fg.beta, self.xp)
            fg.walpha = to_backend(fg.walpha, self.xp)
            fg.wbeta = to_backend(fg.wbeta, self.xp)
            # Coordinates might remain on CPU for plotting if needed, 
            # but usually plotting is done after transferring back.
            # Convert metric terms used in RHS
            fg.sqrt_g = to_backend(fg.sqrt_g, self.xp)
            fg.u1 = to_backend(fg.u1, self.xp)
            fg.u2 = to_backend(fg.u2, self.xp)
            fg.div_u = to_backend(fg.div_u, self.xp)

    def _precompute_static_fields(self, alpha0: float, u0: float):
        """
        Precompute contravariant velocities and divergence for the formulation.
        """
        for fname, fg in self.faces.items():
            # 1. Physical wind (Spherical)
            u_sph, v_sph = self.geometry.solid_body_wind(fg.X, fg.Y, fg.Z, alpha0, u0)
            
            # 2. Contravariant wind
            u1, u2 = self.geometry.compute_contravariant_vel(fg, u_sph, v_sph)
            
            if fg.sqrt_g is None:
                raise ValueError(f"Face {fname} has undefined Jacobian sqrt_g")

            fg.u1 = u1
            fg.u2 = u2
            
            # 3. Divergence of Velocity (Correction Term for DG skew-symmetric)
            # div(u) = (1/sqrt_g) * [ d/da(sqrt_g * u1) + d/db(sqrt_g * u2) ]
            term1 = self.D @ (fg.sqrt_g * u1)
            term2 = (fg.sqrt_g * u2) @ self.D.T
            fg.div_u = (1.0 / fg.sqrt_g) * (term1 + term2)

    def get_initial_condition(self, type: str = "gaussian", 
                              func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
                              h0: float = 1.0, r0: Optional[float] = None) -> np.ndarray:
        """
        Generate an initial scalar field phi.n_vars, 6, N, N).
        
        Args:
            type: "gaussian", "cosine", or "custom" (requires func)
            func: Callable f(lon, lat) -> value. lon/lat in radians.
            h0, r0: Parameters for built-in types.
        """
        if r0 is None: 
            r0 = self.cfg.R / 3.0
        
        center_lon = 0.0
        center_lat = 0.0
        
        # State: (n_vars, 6, N, N)
        if self.cfg.n_vars > 1:
            state = np.zeros((self.cfg.n_vars, 6, self.cfg.N, self.cfg.N))
        else:
            state = np.zeros((1, 6, self.cfg.N, self.cfg.N)) # Keep 4D internally even for n_vars=1 to simplify? 
            # OR we can keep 3D separately.
            # Current request implies (n_vars, 6, ...).
            # But legacy code used (6, ...).
            # Let's standardize on (n_vars, 6, ...)
        
        # ... Geometry calculations done on CPU for simplicity ...
        for i, fname in enumerate(self.topology.FACE_MAP):
            fg = self.faces[fname]
            # Coordinates X,Y,Z might be JAX arrays or Numpy arrays depending on _to_jax
            # Ensure we use numpy version for initial condition math
            X = np.array(fg.X)
            Y = np.array(fg.Y)
            Z = np.array(fg.Z)
            
            lam, theta = self.geometry.lonlat_from_xyz(X, Y, Z)
            
            val = np.zeros_like(lam)
            
            if type == "gaussian":
                sin_c, cos_c = np.sin(center_lat), np.cos(center_lat)
                sin_th, cos_th = np.sin(theta), np.cos(theta)
                cos_sigma = sin_th * sin_c + cos_th * cos_c * np.cos(lam - center_lon)
                r_d = self.cfg.R * np.arccos(np.clip(cos_sigma, -1.0, 1.0))
                val = h0 * np.exp(-(r_d / r0)**2)
            elif type == "cosine":
                sin_c, cos_c = np.sin(center_lat), np.cos(center_lat)
                sin_th, cos_th = np.sin(theta), np.cos(theta)
                cos_sigma = sin_th * sin_c + cos_th * cos_c * np.cos(lam - center_lon)
                r_d = self.cfg.R * np.arccos(np.clip(cos_sigma, -1.0, 1.0))
                val = np.where(r_d < r0, 0.5 * h0 * (1.0 + np.cos(np.pi * r_d / r0)), 0.0)
            elif type == "custom":
                if func is None:
                    raise ValueError("must provide 'func' argument when type='custom'")
                val = func(lam, theta)
            else:
                # If user provided func but didn't specify type="custom", we can infer it
                if func is not None:
                    val = func(lam, theta)
                else:
                    raise ValueError(f"Unknown initial condition type: {type}")
            
            # Broadcast val to all vars by default?
            # Or just set var 0 and let user handle others?
            # For backward compatibility, set all vars to the same IC or just use "custom" with variable awareness?
            # The prompt implies initializing system.
            # We'll set all vars to the same unless customized later.
            state[:, i, :, :] = val
        
        # Squeeze if n_vars=1? No, prompt says 'Use (n_vars, 6, N, N)'.
        # However, to maintain some compatibility with scalar code external to this class?
        # The prompt says: "Refactor... to support N_vars >= 1".
        
        if self.use_jax:
             from cubed_sphere import backend
             return backend.to_backend(state, self.xp)
             
        return state

    def compute_rhs(self, t: float, global_phi: np.ndarray) -> np.ndarray:
        """ Dispatcher for RHS computation """
        if self.use_jax:
            return self._compute_rhs_jax(t, global_phi)
        return self._compute_rhs_numpy(t, global_phi)

    def _compute_rhs_jax(self, t: float, global_phi: np.ndarray) -> np.ndarray:
        """ Functional JAX implementation of RHS """
        xp = self.xp
        rhs_list = []
        map_factor = 4.0 / np.pi
        
        def da(F): return self.D @ F * map_factor
        def db(F): return F @ self.D.T * map_factor # Verify D.T behavior in JAX
        
        # In JAX, we cannot mutate penalty array. 
        # We calculate terms for each face and stack them.

        for i, fname in enumerate(self.topology.FACE_MAP):
            fg = self.faces[fname]
            phi = global_phi[..., i, :, :] # (n_vars, N, N)
            
            # --- 1. Volume Integral ---
            flux = (1.0 / fg.sqrt_g) * (da(fg.sqrt_g * fg.u1 * phi) + db(fg.sqrt_g * fg.u2 * phi))
            advect = fg.u1 * da(phi) + fg.u2 * db(phi)
            correction = phi * fg.div_u
            skew_div = 0.5 * (flux + advect - correction)
            
            # --- 2. Surface Integral (SAT Penalty) ---
            # We must compute penalty contribution from all 4 sides essentially "inline" or functional
            
            # Helper: SAT penalty
            def sat(vn, q_in, q_out, w_metric):
                flux_diff = 0.5 * (vn - xp.abs(vn)) * (q_in - q_out)
                return flux_diff / w_metric

            # Calculate penalties for the 4 edges of THIS face
            # Note: get_neighbor_data now returns (..., edge_len)
            
            p_west = sat(-fg.u1[0, :], phi[..., 0, :], self.topology.get_neighbor_data(global_phi, i, 0), fg.sqrt_g[0, :] * fg.walpha[0])
            p_east = sat(fg.u1[-1, :], phi[..., -1, :], self.topology.get_neighbor_data(global_phi, i, 1), fg.sqrt_g[-1, :] * fg.walpha[-1])
            p_south = sat(-fg.u2[:, 0], phi[..., :, 0], self.topology.get_neighbor_data(global_phi, i, 2), fg.sqrt_g[:, 0] * fg.wbeta[0])
            p_north = sat(fg.u2[:, -1], phi[..., :, -1], self.topology.get_neighbor_data(global_phi, i, 3), fg.sqrt_g[:, -1] * fg.wbeta[-1])
            
            # Construct penalty field (shape n_vars, N, N)
            # Start with zeros
            pen = xp.zeros_like(phi)
            # Add updates. Since phi is (..., N, N), we use .at
            pen = pen.at[..., 0, :].add(p_west)
            pen = pen.at[..., -1, :].add(p_east)
            pen = pen.at[..., :, 0].add(p_south)
            pen = pen.at[..., :, -1].add(p_north)
            
            rhs_i = -skew_div + pen
            rhs_list.append(rhs_i)
        
        # Stack on axis 1 (face dim)
        return xp.stack(rhs_list, axis=1)

    def _compute_rhs_numpy(self, t: float, global_phi: np.ndarray) -> np.ndarray:
        """
        Compute d(phi)/dt RHS = - (Skew-Symmetric Divergence) + (SAT Penalty)
        Optimized for NumPy (In-Place Mutation)
        """
        # Note: t is currently unused for autonomous advection
        rhs = np.zeros_like(global_phi)
        map_factor = 4.0 / np.pi
        
        # Helper: element-wise derivatives mapped to reference [-1,1]
        def da(F): return self.D @ F * map_factor
        def db(F): return F @ self.D.T * map_factor
        
        # Helper: SAT penalty
        def sat(vn, q_in, q_out, w_metric):
            # Only penalize inflow (Vn < 0 relative to outward normal?)
            # Usually SAT: tau * (Flux_internal - Flux_numerical)
            # Here: 0.5 * (vn - |vn|) * (q_in - q_out) -> 
            # If vn > 0 (outflow), term is 0. 
            # If vn < 0 (inflow), term is vn * (q_in - q_out).
            flux_diff = 0.5 * (vn - np.abs(vn)) * (q_in - q_out)
            return flux_diff / w_metric

        for i, fname in enumerate(self.topology.FACE_MAP):
            fg = self.faces[fname]
            phi = global_phi[..., i, :, :] # (n_vars, N, N) or (N, N)
            
            if fg.sqrt_g is None or fg.u1 is None or fg.u2 is None or fg.div_u is None:
                raise RuntimeError(f"Face {fname} not fully initialized.")

            # --- 1. Volume Integral (Skew-Symmetric Form) ---
            # Flux Term: (1/J) * div(J * u * phi)
            # broadcasting D (N,N) @ F (..., N, N) works in numpy
            flux = (1.0 / fg.sqrt_g) * (da(fg.sqrt_g * fg.u1 * phi) + db(fg.sqrt_g * fg.u2 * phi))
            
            # Advective Term: u * grad(phi)
            advect = fg.u1 * da(phi) + fg.u2 * db(phi)
            
            # Correction Term: phi * div(u)
            correction = phi * fg.div_u
            
            # Skew-Symmetric Average
            skew_div = 0.5 * (flux + advect - correction)
            
            # --- 2. Surface Integral (SAT Penalty) ---
            penalty = np.zeros_like(phi)

            # West (Alpha=-1) -> Vn = -u1
            q_out = self.topology.get_neighbor_data(global_phi, i, 0)
            penalty[..., 0, :] += sat(-fg.u1[0, :], phi[..., 0, :], q_out, fg.sqrt_g[0, :] * fg.walpha[0])
            
            # East (Alpha=1) -> Vn = u1
            q_out = self.topology.get_neighbor_data(global_phi, i, 1)
            penalty[..., -1, :] += sat(fg.u1[-1, :], phi[..., -1, :], q_out, fg.sqrt_g[-1, :] * fg.walpha[-1])
            
            # South (Beta=-1) -> Vn = -u2
            q_out = self.topology.get_neighbor_data(global_phi, i, 2)
            penalty[..., :, 0] += sat(-fg.u2[:, 0], phi[..., :, 0], q_out, fg.sqrt_g[:, 0] * fg.wbeta[0])
            
            # North (Beta=1) -> Vn = u2
            q_out = self.topology.get_neighbor_data(global_phi, i, 3)
            penalty[..., :, -1] += sat(fg.u2[:, -1], phi[..., :, -1], q_out, fg.sqrt_g[:, -1] * fg.wbeta[-1])
            
            rhs[..., i, :, :] = -skew_div + penalty
            
        return rhs

    def _step_jax(self, state, dt):
        """Standard LSRK5 step (functional style for JAX JIT)"""
        du = self.xp.zeros_like(state)
        local_state = state
        
        # Unroll loop for JIT trace
        for k in range(5):
            rhs = self._compute_rhs_jax(0.0, local_state) # t is unused
            
            # du = a[k] * du + dt * rhs
            du = self.rk_a[k] * du + dt * rhs
            
            # state = state + b[k] * du
            local_state = local_state + self.rk_b[k] * du
            
        return local_state

    def solve(self, t_span: Tuple[float, float], initial_state: np.ndarray) -> np.ndarray:
        """
        Run simulation from t_span[0] to t_span[1] starting with initial_state.
        Dispatches to JAX or NumPy implementation.
        """
        if self.use_jax:
             return self._solve_jax(t_span, initial_state)
        
        # === NumPy Implementation (In-Place Optimized) ===
        t_start, t_end = t_span
        current_time = t_start
        
        # Estimate dt (simple constant dt based on CFL)
        dt_est = (self.cfg.CFL / self.cfg.u0) * (2 / self.cfg.N**2)
        
        print(f"=== Starting Simulation (NumPy) ===")
        print(f"N={self.cfg.N}, CFL={self.cfg.CFL}, dt={dt_est:.5f}, T_span={t_span}")
        
        state = initial_state.copy()
        du = np.zeros_like(state)
        step_count = 0
        epsilon = 1e-12
        
        while current_time < t_end - epsilon:
            remaining = t_end - current_time
            if remaining < dt_est:
                step_dt = remaining
            else:
                step_dt = dt_est
            
            for k in range(5):
                rhs = self._compute_rhs_numpy(current_time, state)
                
                # Update du (Residual) In-Place
                if self.rk_a[k] == 0.0:
                    np.multiply(rhs, step_dt, out=du)
                else:
                    du *= self.rk_a[k]
                    rhs *= step_dt
                    du += rhs
                
                # Update State In-Place
                state += self.rk_b[k] * du
            
            current_time += step_dt
            step_count += 1
            
            # Print occasionally (optional check to avoid spam)
            if step_count % 50 == 0:
                 print(f"Step {step_count}: t={current_time:.4f}")
                
        print("=== Simulation Complete ===")
        return state

    def _solve_jax(self, t_span: Tuple[float, float], initial_state: Any) -> Any:
        """
        JAX implementation using lax.scan for bulk steps and one extra step for residual.
        This avoids Python loop overhead and enables XLA optimization for the time loop.
        """
        import jax.lax as lax
        
        t_start, t_end = t_span
        duration = t_end - t_start
        
        # Constant dt
        dt_est = (self.cfg.CFL / self.cfg.u0) * (2 / self.cfg.N**2)
        
        # Calculate steps
        n_steps = int(duration // dt_est)
        residual = duration - n_steps * dt_est
        
        print(f"=== Starting Simulation (JAX) ===")
        print(f"N={self.cfg.N}, CFL={self.cfg.CFL}, dt={dt_est:.5f}")
        print(f"Total Duration={duration:.5f} -> Full Steps={n_steps}, Residual={residual:.5e}")
        
        # 1. Bulk steps using lax.scan (Compiled Loop)
        # Note: self._jit_step is already JIT-compiled. 
        # We wrap it in a scanner function.
        def scan_body(carry, _):
            s = carry
            # Fixed dt for bulk steps
            s_new = self._jit_step(s, dt_est)
            return s_new, None

        state = initial_state
        
        if n_steps > 0:
            # Execute n_steps
            state, _ = lax.scan(scan_body, state, None, length=n_steps)
            
        # 2. Residual step (if needed)
        if residual > 1e-12:
            print(f"Performing residual step dt={residual:.5e}")
            state = self._jit_step(state, residual)

        print("=== Simulation Complete (JAX) ===")
        return state 
        # API contract says "return final_state". If user wants numpy, they can convert.
        # But 'solve' usually returns framework-native output.
        return state

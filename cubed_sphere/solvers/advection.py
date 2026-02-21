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
    dt: Optional[float] = None # Time step (Optional)

class CubedSphereAdvectionSolver(BaseSolver):
    """
    Discontinuous Galerkin solver for the advection equation on the Cubed Sphere.
    Supports both NumPy (CPU) and JAX (GPU/TPU) backends.
    """
    def __init__(self, config: AdvectionConfig):
        super().__init__(dataclasses.asdict(config))
        self.cfg = config
        self.N = config.N  # Required for BaseSolver.compute_safe_dt safety check
        self.R = config.R  # Explicitly store R for consistency
        
        # Backend Setup
        from cubed_sphere import backend
        self.xp = backend.get_backend(config.backend)
        self.use_jax = (config.backend == 'jax')
        
        self.topology = CubedSphereTopology()
        self.geometry = CubedSphereEquiangular(config.R)
        self.num_nodes = config.N + 1
        self.D_cpu = lgl_diff_matrix(self.num_nodes) # Keep CPU copy for init
        self.D = self.D_cpu # Initialize D for precomputation (NumPy)
        
        # Initialize Grid Faces
        self.faces: Dict[str, FaceGrid] = {}
        for fname in self.topology.FACE_MAP:
            self.faces[fname] = self.geometry.generate_face(self.num_nodes, fname)
            
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

    def _precompute_static_fields(self, alpha0: float, u0: float):
        """
        Precompute static geometric fields (metrics) and wind velocity.
        """
        # Scale for spectral differentiation (from [-1, 1] to [-pi/4, pi/4])
        scale = 4.0 / np.pi

        for fname, fg in self.faces.items():
            # --- 1. Compute Metric Tensor Components (Covariant g_ij) ---
            # Used for calculating physical velocity magnitude |u|^2 = g_ij u^i u^j
            
            # Basis vectors (Tangent vectors): g_1 = dX/dAlpha, g_2 = dX/dBeta
            # D matrix represents d/dZeta. d/dAlpha = (dZeta/dAlpha) * d/dZeta = (4/pi) * D
            
            # Derivatives wrt Alpha (use pre-computed D_cpu for initialization)
            g1_x = self.D_cpu @ fg.X * scale
            g1_y = self.D_cpu @ fg.Y * scale
            g1_z = self.D_cpu @ fg.Z * scale
            
            # Derivatives wrt Beta
            g2_x = fg.X @ self.D_cpu.T * scale
            g2_y = fg.Y @ self.D_cpu.T * scale
            g2_z = fg.Z @ self.D_cpu.T * scale
            
            # Metric Components (Dot products of basis vectors)
            fg.g_11 = g1_x*g1_x + g1_y*g1_y + g1_z*g1_z
            fg.g_22 = g2_x*g2_x + g2_y*g2_y + g2_z*g2_z
            fg.g_12 = g1_x*g2_x + g1_y*g2_y + g1_z*g2_z

            # --- 2. Physical wind (Spherical) ---
            u_sph, v_sph = self.geometry.solid_body_wind(fg.X, fg.Y, fg.Z, alpha0, u0)
            
            # --- 3. Contravariant wind ---
            u1, u2 = self.geometry.compute_contravariant_vel(fg, u_sph, v_sph)
            
            if fg.sqrt_g is None:
                raise ValueError(f"Face {fname} has undefined Jacobian sqrt_g")

            fg.u1 = u1
            fg.u2 = u2
            
            # --- 4. Divergence (Correction Term) ---
            term1 = self.D_cpu @ (fg.sqrt_g * u1)
            term2 = (fg.sqrt_g * u2) @ self.D_cpu.T
            fg.div_u = (1.0 / fg.sqrt_g) * (term1 + term2)

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
            
            # Convert Metric Arrays
            fg.sqrt_g = to_backend(fg.sqrt_g, self.xp)
            fg.g_11 = to_backend(fg.g_11, self.xp)
            fg.g_12 = to_backend(fg.g_12, self.xp)
            fg.g_22 = to_backend(fg.g_22, self.xp)
            
            # Convert Velocity Fields
            fg.u1 = to_backend(fg.u1, self.xp)
            fg.u2 = to_backend(fg.u2, self.xp)
            fg.div_u = to_backend(fg.div_u, self.xp)

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
        
        # Determine grid size (Nodes = N+1 for LGL)
        num_nodes = self.cfg.N + 1
        
        # State: (n_vars, 6, N+1, N+1)
        if self.cfg.n_vars > 1:
            state = np.zeros((self.cfg.n_vars, 6, num_nodes, num_nodes))
        else:
            state = np.zeros((1, 6, num_nodes, num_nodes)) # Keep 4D internally even for n_vars=1 to simplify? 
            # OR we can keep 3D separately.
            # Current request implies (n_vars, 6, ...).
            # But legacy code used (6, ...).
            # Let's standardize on (n_vars, 6, ...)
        
        # ... Geometry calculations done on CPU for simplicity ...
        
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
            
            p_west = sat(-fg.u1[0, :], phi[..., 0, :], self.topology.get_neighbor_data(global_phi, i, 0), fg.walpha[0])
            p_east = sat(fg.u1[-1, :], phi[..., -1, :], self.topology.get_neighbor_data(global_phi, i, 1), fg.walpha[-1])
            p_south = sat(-fg.u2[:, 0], phi[..., :, 0], self.topology.get_neighbor_data(global_phi, i, 2), fg.wbeta[0])
            p_north = sat(fg.u2[:, -1], phi[..., :, -1], self.topology.get_neighbor_data(global_phi, i, 3), fg.wbeta[-1])
            
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
            penalty[..., 0, :] += sat(-fg.u1[0, :], phi[..., 0, :], q_out, fg.walpha[0])
            
            # East (Alpha=1) -> Vn = u1
            q_out = self.topology.get_neighbor_data(global_phi, i, 1)
            penalty[..., -1, :] += sat(fg.u1[-1, :], phi[..., -1, :], q_out, fg.walpha[-1])
            
            # South (Beta=-1) -> Vn = -u2
            q_out = self.topology.get_neighbor_data(global_phi, i, 2)
            penalty[..., :, 0] += sat(-fg.u2[:, 0], phi[..., :, 0], q_out, fg.wbeta[0])
            
            # North (Beta=1) -> Vn = u2
            q_out = self.topology.get_neighbor_data(global_phi, i, 3)
            penalty[..., :, -1] += sat(fg.u2[:, -1], phi[..., :, -1], q_out, fg.wbeta[-1])
            
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

    def _get_max_wave_speed(self, state=None) -> float:
        """
        Calculates the maximum local wave speed (sqrt(g_ij u^i u^j)) on the grid.
        Uses precomputed metric tensors for full vectorization and JAX compatibility.
        
        Args:
            state: Unused, but kept for API compatibility with BaseSolver.
        """
        local_max_speeds = []
        
        for fg in self.faces.values():
            # Calculate physical velocity squared magnitude using precomputed metrics
            # |v|^2 = g_11*(u1)^2 + 2*g_12*u1*u2 + g_22*(u2)^2
            v_sq = (fg.g_11 * fg.u1**2 + 
                    2.0 * fg.g_12 * fg.u1 * fg.u2 + 
                    fg.g_22 * fg.u2**2)
            
            # Use self.xp (numpy or jax.numpy) for array operations
            # Clamp negative values (numerical noise) before sqrt
            v_mag = self.xp.sqrt(self.xp.maximum(v_sq, 0.0))
            local_max_speeds.append(self.xp.max(v_mag))

        # Return global maximum
        return self.xp.max(self.xp.array(local_max_speeds))

    def step(self, t: float, state: np.ndarray, dt: float = None) -> np.ndarray:
        """
        Advances the state by one time step.
        Args:
            t (float): Current simulation time.
            state (array): Current state array.
            dt (float, optional): Time step size. If None, use internal default.
        Returns:
            new_state (array): The state after one step.
        """
        if dt is None:
            # Use compute_safe_dt from BaseSolver
            dt = self.compute_safe_dt(state, self.cfg.CFL)

        if self.use_jax:
             return self._jit_step(state, dt)

        # NumPy implementation
        local_state = state.copy()
        du = np.zeros_like(local_state)
        
        for k in range(5):
            rhs = self._compute_rhs_numpy(t, local_state)
            
            # Update du (Residual) In-Place
            if self.rk_a[k] == 0.0:
                np.multiply(rhs, dt, out=du)
            else:
                du *= self.rk_a[k]
                rhs *= dt
                du += rhs
            
            # Update State In-Place
            local_state += self.rk_b[k] * du
            
        return local_state

    def solve(self, t_span: Tuple[float, float], initial_state: np.ndarray, callbacks: List[Any] = None) -> np.ndarray:
        """
        Run simulation from t_span[0] to t_span[1] starting with initial_state.
        Dispatches to JAX or NumPy implementation.
        """
        if self.use_jax:
             return self._solve_jax(t_span, initial_state, callbacks)
        
        # === NumPy Implementation ===
        t_start, t_end = t_span
        current_time = t_start
        
        # Estimate dt: Use config.dt if provided, else use heuristic
        if self.cfg.dt is not None:
            dt_est = self.cfg.dt
        else:
            # Heuristic for DG on Sphere: dt ~ CFL * dx_min / u_max
            # Approximate dx ~ R / N (actually smaller near corners, scaling with 1/N^2 for stability)
            # Factor of R was missing in original heuristic if u0 is physical velocity (m/s)
            dt_est = (self.cfg.CFL * self.cfg.R / self.cfg.u0) * (1.0 / self.cfg.N**2)

        print(f"=== Starting Simulation (NumPy) ===")
        print(f"N={self.cfg.N}, CFL={self.cfg.CFL}, dt={dt_est:.5f}, T_span={t_span}")
        
        state = initial_state.copy()
        step_count = 0
        epsilon = 1e-12
        
        # Initial Callback
        if callbacks:
            for cb in callbacks:
                cb(current_time, state)
        
        # Determine print interval (try to print ~20 times per run or every 1000 steps)
        total_steps_est = (t_end - t_start) / dt_est
        print_interval = max(50, int(total_steps_est / 20))
        
        while current_time < t_end - epsilon:
            remaining = t_end - current_time
            if remaining < dt_est:
                step_dt = remaining
            else:
                step_dt = dt_est
            
            state = self.step(current_time, state, step_dt)
            
            current_time += step_dt
            step_count += 1
            
            # Callback
            if callbacks:
                for cb in callbacks:
                    cb(current_time, state)
            
            # Print occasionally
            if step_count % print_interval == 0:
                 print(f"Step {step_count}: t={current_time:.4f}")
                
        print("=== Simulation Complete ===")
        return state

    def _solve_jax(self, t_span: Tuple[float, float], initial_state: Any, callbacks: List[Any] = None) -> Any:
        """
        JAX implementation using lax.scan for bulk steps.
        If callbacks are present, uses 'Chunked Stepping' to allow CPU interrupts for I/O.
        """
        import jax.lax as lax
        import numpy as np
        
        t_start, t_end = t_span
        duration = t_end - t_start
        
        # Estimate dt
        if self.cfg.dt is not None:
            dt_est = self.cfg.dt
        else:
             dt_est = (self.cfg.CFL * self.cfg.R / self.cfg.u0) * (1.0 / self.cfg.N**2)

        def scan_body(carry, _):
            s = carry
            s_new = self._jit_step(s, dt_est)
            return s_new, None

        state = initial_state
        current_time = t_start
        
        # --- Strategy Selection ---
        if not callbacks:
            # OPTION A: Maximum Speed (No Callbacks)
            n_steps = int(duration // dt_est)
            residual = duration - n_steps * dt_est
            
            print(f"=== Starting Simulation (JAX - Fast Mode) ===")
            print(f"Total Steps={n_steps}, Residual={residual:.5e}")
            
            if n_steps > 0:
                state, _ = lax.scan(scan_body, state, None, length=n_steps)
            if residual > 1e-12:
                state = self._jit_step(state, residual)
        else:
            # OPTION B: Chunked Stepping (With I/O)
            # Infer save interval from callbacks? Hard to know exactly.
            # Default to 0.05 sim seconds or 100 steps
            chunk_dt = 100 * dt_est 
            # Try to get interval from monitor if available
            for cb in callbacks:
                if hasattr(cb, 'save_interval'):
                    chunk_dt = min(chunk_dt, cb.save_interval)
            
            # Ensure chunk_dt is at least 1 step
            if chunk_dt < dt_est: chunk_dt = dt_est * 10
            
            print(f"=== Starting Simulation (JAX - Chunked Mode) ===")
            print(f"Chunk Size ~ {chunk_dt:.4f}s")
             
            # Initial Callback
            current_state_np = np.array(state)
            for cb in callbacks: cb(current_time, current_state_np)

            # Outer Python Loop
            while current_time < t_end - 1e-9:
                # Determine next stop
                next_target = min(current_time + chunk_dt, t_end)
                chunk_duration = next_target - current_time
                
                n_substeps = int(chunk_duration // dt_est)
                if n_substeps < 1 and chunk_duration > 1e-12:
                     # Just residual step
                     n_substeps = 0
                     residual = chunk_duration
                else:
                     residual = chunk_duration - n_substeps * dt_est
                
                # Run JIT Kernel
                if n_substeps > 0:
                    state, _ = lax.scan(scan_body, state, None, length=n_substeps)
                if residual > 1e-12:
                    state = self._jit_step(state, residual)
                
                current_time = next_target
                
                # Sync & Callback
                # Note: `state` is on GPU. converting to np array forces sync.
                current_state_np = np.array(state)
                for cb in callbacks:
                    cb(current_time, current_state_np)
                    
        print("=== Simulation Complete (JAX) ===")
        return state
            
        # 2. Residual step (if needed)
        if residual > 1e-12:
            print(f"Performing residual step dt={residual:.5e}")
            state = self._jit_step(state, residual)

        print("=== Simulation Complete (JAX) ===")
        return state 
        # API contract says "return final_state". If user wants numpy, they can convert.
        # But 'solve' usually returns framework-native output.
        return state

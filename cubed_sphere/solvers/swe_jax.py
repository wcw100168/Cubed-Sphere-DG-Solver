import jax
import jax.numpy as jnp
from jax import jit, lax, Array
import numpy as np
import dataclasses
from functools import partial
from typing import Dict, Tuple, List, Any, Optional, Union
from cubed_sphere.solvers.base import BaseSolver
from cubed_sphere.geometry.grid import CubedSphereTopology, CubedSphereEquiangular, FaceGrid, GlobalGrid, FaceMetrics, freeze_grid
from cubed_sphere.numerics.spectral import lgl_diff_matrix, lgl_nodes_weights
from cubed_sphere.physics.initialization import get_initial_state
from numpy.polynomial.legendre import Legendre
from jax import lax
import math

# Physics Constants
def _get_dtype():
    from jax import config
    return jnp.float64 if config.read("jax_enable_x64") else jnp.float32

# Defer creation to ensure we catch the config, or use property?
# We'll use a lazy approach or just init them here.
# NOTE: User should configure JAX before importing if possible, 
# but we can also re-cast inside the class to be safe.
EARTH_RADIUS = 6.37122e6
OMEGA = 7.292e-5
GRAVITY = 9.80616

def compute_bv_filter_matrix(N: int, strength=36.0, order=16) -> np.ndarray:
    """
    Computes the Boyd-Vandeven filter matrix F = V * Lambda * V^-1.
    """
    num_nodes = N + 1
    nodes, _ = lgl_nodes_weights(-1.0, 1.0, num_nodes)
    V = np.zeros((num_nodes, num_nodes))
    for j in range(num_nodes):
        Pj = Legendre.basis(j)
        V[:, j] = Pj(nodes)
        
    sigma = np.zeros(num_nodes)
    for k in range(num_nodes):
        if N > 0:
            sigma[k] = np.exp(-strength * (float(k)/N)**order)
        else:
            sigma[k] = 1.0
        
    Lambda = np.diag(sigma)
    V_inv = np.linalg.inv(V)
    F = V @ Lambda @ V_inv
    return F

class CubedSphereSWEJax(BaseSolver):
    """
    Shallow Water Equations solver on Cubed Sphere using JAX.
    
    Implements the Vector Invariant Formulation with Rusanov Fluxes.
    Matches the numerical method of CubedSphereSWENumpy but optimized for
    accelerator execution (GPU/TPU) via JIT compilation and `jax.lax.scan`.
    
    Architecture:
    - Stateless `GlobalGrid` design (metric terms passed as arguments).
    - Hybrid Initialization: Uses NumPy for complex geometry setup, then freezes to JAX.
    - Dual-Path Execution: Supports both interactive Python loops (for debugging/callbacks)
      and fused XLA kernels (for high-performance simulation).
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.N = config.get('N', 32)
        
        # Resolve dtype and cast constants
        self.dtype = _get_dtype()
        # Ensure constants are Arrays with correct dtype
        self.R = jnp.array(config.get('R', EARTH_RADIUS), dtype=self.dtype)
        self.omega = jnp.array(OMEGA, dtype=self.dtype)
        self.gravity = jnp.array(GRAVITY, dtype=self.dtype)
        
        # 1. Topology & Geometry
        self.topology = CubedSphereTopology()
        
        # We perform initialization using NumPy for geometry/topo 
        # and then convert the result to JAX arrays.
        self.num_nodes = self.N + 1
        
        # Diff Matrix (NumPy init)
        D_np = lgl_diff_matrix(self.num_nodes)
        self.D = jnp.array(D_np, dtype=self.dtype) # JAX Array
        
        # Use simple float for geometry init, cast later
        self.geometry = CubedSphereEquiangular(float(self.R))
        
        # 2. Operators - Filter
        # Compute in Numpy, convert to JAX
        filter_np = compute_bv_filter_matrix(self.N, strength=36.0, order=16)
        self.filter_matrix = jnp.array(filter_np, dtype=self.dtype)
        
        # RK5 Coefficients
        self.rk_a = jnp.array([0.0, 
                              -567301805773.0/1357537059087.0, 
                              -2404267990393.0/2016746695238.0, 
                              -3550918686646.0/2091501179385.0, 
                              -1275806237668.0/842570457699.0], dtype=self.dtype)
        self.rk_b = jnp.array([1432997174477.0/9575080441755.0, 
                              5161836677717.0/13612068292357.0, 
                              1720146321549.0/2090206949498.0, 
                              3134564353537.0/4481467310338.0, 
                              2277821191437.0/14882151754819.0], dtype=self.dtype)
        
        # 3. Initialize Faces and Extended Metrics
        # We need to compute metrics, then store them in a way accessible to JAX.
        # Ideally, we put them into JAX arrays inside FaceGrid, 
        # but FaceGrid is a dataclass. Let's populate it with jnp.arrays.
        
        self.faces = {}
        for fname in self.topology.FACE_MAP:
            # Generate (NumPy)
            fg = self.geometry.generate_face(self.num_nodes, fname) 
            # Extend (NumPy -> JAX inside)
            self._compute_extended_metrics(fg, D_np) 
            self.faces[fname] = fg
            
        # 4. Freeze Grid for JAX (NumPy -> GlobalGrid PyTree via freeze_grid)
        self.grid_metrics = freeze_grid(self.faces)

    def get_initial_condition(self):
        # Convert JAX faces back to NumPy for initialization logic
        faces_np = {}
        for fname, fg in self.faces.items():
            fg_np = dataclasses.replace(fg)
            # Helper to safely convert if it's jax array
            def to_np(arr):
                return np.array(arr) if hasattr(arr, 'device_buffer') or str(type(arr)).find('jax') != -1 else arr
            
            fg_np.g1_vec = to_np(fg.g1_vec)
            fg_np.g2_vec = to_np(fg.g2_vec)
            fg_np.g_inv = to_np(fg.g_inv)
            fg_np.g_ij = to_np(fg.g_ij)
            fg_np.sqrt_g = to_np(fg.sqrt_g)
            fg_np.f_coriolis = to_np(fg.f_coriolis)
            
            # --- MANUALLY COPY LON/LAT FROM DYNAMIC ATTRIBUTES ---
            if hasattr(fg, 'lon'):
                 setattr(fg_np, 'lon', to_np(fg.lon)) # Use setattr for dynamic
            if hasattr(fg, 'lat'):
                 setattr(fg_np, 'lat', to_np(fg.lat))
            
            faces_np[fname] = fg_np

        case_id = self.config.get('initial_condition', 6)
        if isinstance(case_id, int):
            case_id = f"case{case_id}"
            
        # Call shared logic
        state_np = get_initial_state(
            config=self.config, 
            faces=faces_np, 
            case_type=case_id, 
            **self.config
        )
        
        return jax.device_put(state_np)

    def _compute_extended_metrics(self, fg: FaceGrid, D_np: np.ndarray):
        """
        Compute metrics using NumPy, then convert to JAX.
        Uses the provided D_np (numpy diff matrix) to avoid JAX/Numpy mixing errors.
        """
        scale = 4.0 / np.pi
        
        # NumPy Operations
        g1_x = D_np @ fg.X * scale
        g2_x = fg.X @ D_np.T * scale 
        g1_y = D_np @ fg.Y * scale
        g2_y = fg.Y @ D_np.T * scale
        g1_z = D_np @ fg.Z * scale
        g2_z = fg.Z @ D_np.T * scale
        
        g1_vec = np.stack([g1_x, g1_y, g1_z], axis=-1)
        g2_vec = np.stack([g2_x, g2_y, g2_z], axis=-1)
        
        g11 = np.sum(g1_vec * g1_vec, axis=-1)
        g22 = np.sum(g2_vec * g2_vec, axis=-1)
        g12 = np.sum(g1_vec * g2_vec, axis=-1)
        
        det = g11*g22 - g12**2
        inv_det = 1.0 / det
        
        g_inv = np.zeros(g11.shape + (2, 2))
        g_inv[..., 0, 0] = g22 * inv_det
        g_inv[..., 1, 1] = g11 * inv_det
        g_inv[..., 0, 1] = -g12 * inv_det
        g_inv[..., 1, 0] = -g12 * inv_det
        
        # Calculate g_ij (Covariant)
        g_ij = np.zeros(g11.shape + (2, 2))
        g_ij[..., 0, 0] = g11
        g_ij[..., 1, 1] = g22
        g_ij[..., 0, 1] = g12
        g_ij[..., 1, 0] = g12
        
        # Explicit casts to self.dtype
        fg.g1_vec = jnp.array(g1_vec, dtype=self.dtype)
        fg.g2_vec = jnp.array(g2_vec, dtype=self.dtype)
        fg.g_inv = jnp.array(g_inv, dtype=self.dtype)
        fg.g_ij  = jnp.array(g_ij, dtype=self.dtype)
        fg.sqrt_g = jnp.array(fg.sqrt_g, dtype=self.dtype) # Use existing analytical sqrt_g
        fg.f_coriolis = jnp.array(f_coriolis, dtype=self.dtype)# Convert to JAX and store in fg
        fg.g1_vec = jnp.array(g1_vec)
        fg.g2_vec = jnp.array(g2_vec)
        fg.g_inv = jnp.array(g_inv)
        fg.g_ij  = jnp.array(g_ij)
        fg.sqrt_g = jnp.array(fg.sqrt_g) # Use existing analytical sqrt_g
        fg.f_coriolis = jnp.array(f_coriolis)
        
        # Store lon/lat for utility (NumPy or JAX, doesn't matter for JIT as config)
        fg.lon = lam
        fg.lat = theta

    def _get_boundary_flux_data_jax(self, global_state, face_idx, side, grid: GlobalGrid):
        """JAX version of _get_boundary_flux_data."""
        # This needs to be traceable.
        # Accessing `self.topology.CONN_TABLE` is fine as it's static dict.
        # Accessing `grid` NamedTuple via `grid.faces[...]` is fine.
        
        nbr_face, nbr_side, swap, reverse = self.topology.CONN_TABLE[(face_idx, side)]
        nbr_metrics: FaceMetrics = grid.faces[nbr_face]
        
        # global_state: (3, 6, N, N)
        nbr_U = global_state[:, nbr_face, :, :] # (3, N, N)
        
        # Slicing
        # Static control flow (if/else) on `nbr_side` is fine since it's unrolled per face/side loop iteration.
        
        if nbr_side == 0:   
            val = nbr_U[:, 0, :]   # Alpha=0 (West)
        elif nbr_side == 1: 
            val = nbr_U[:, -1, :]  # Alpha=-1 (East)
        elif nbr_side == 2: 
            val = nbr_U[:, :, 0]   # Beta=0 (South)
        elif nbr_side == 3: 
            val = nbr_U[:, :, -1]  # Beta=-1 (North)
            
        m_nb = val[0]
        u1_nb = val[1]
        u2_nb = val[2]
        
        # Reverse
        # `reverse` is python bool from table.
        if reverse:
            m_nb = m_nb[::-1]; u1_nb = u1_nb[::-1]; u2_nb = u2_nb[::-1]
            
        # Vectors
        g1_full, g2_full = nbr_metrics.g1_vec, nbr_metrics.g2_vec
        
        if nbr_side == 0:   g1, g2 = g1_full[0, :], g2_full[0, :]
        elif nbr_side == 1: g1, g2 = g1_full[-1, :], g2_full[-1, :]
        elif nbr_side == 2: g1, g2 = g1_full[:, 0], g2_full[:, 0]
        elif nbr_side == 3: g1, g2 = g1_full[:, -1], g2_full[:, -1]
        
        if reverse:
            g1 = g1[::-1]; g2 = g2[::-1]

        # Cartesian Reconstruction
        g11 = jnp.sum(g1*g1, axis=1); g12 = jnp.sum(g1*g2, axis=1); g22 = jnp.sum(g2*g2, axis=1)
        det = g11*g22 - g12**2
        
        u1_contra = (g22 * u1_nb - g12 * u2_nb) / det
        u2_contra = (g11 * u2_nb - g12 * u1_nb) / det
        
        Vx = u1_contra * g1[:,0] + u2_contra * g2[:,0]
        Vy = u1_contra * g1[:,1] + u2_contra * g2[:,1]
        Vz = u1_contra * g1[:,2] + u2_contra * g2[:,2]
        
        return m_nb, jnp.stack([Vx, Vy, Vz], axis=-1)

    @partial(jit, static_argnums=(0,))
    def _compute_rhs_core(self, t, global_state, grid: GlobalGrid, D):
        dtype = global_state.dtype
        rhs = jnp.zeros_like(global_state)
        # Force cast scalar
        scale = jnp.array(4.0 / np.pi, dtype=dtype)
        
        def da(F): return D @ F * scale
        def db(F): return F @ D.T * scale
        
        # We loop over faces 0..5. Since 6 is small, unrolling is efficient.
        for i in range(6):
            fg_m: FaceMetrics = grid.faces[i]
            
            # Unpack Variables (Layout: 3, 6, N, N)
            m = global_state[0, i]
            u1_cov = global_state[1, i]
            u2_cov = global_state[2, i]
            
            h = m / fg_m.sqrt_g
            g_inv = fg_m.g_inv
            
            # 1. Contravariant
            u1_con = g_inv[...,0,0]*u1_cov + g_inv[...,0,1]*u2_cov
            u2_con = g_inv[...,1,0]*u1_cov + g_inv[...,1,1]*u2_cov
            
            # 2. Vorticity & Energy
            vort = (1.0 / fg_m.sqrt_g) * (da(u2_cov) - db(u1_cov))
            abs_vort = vort + fg_m.f_coriolis
            
            KE = 0.5 * (u1_cov * u1_con + u2_cov * u2_con)
            Phi = self.gravity * h # Use class-level gravity.,1,0]*u1_cov + g_inv[...,1,1]*u2_cov
            
            # 2. Vorticity & Energy
            vort = (1.0 / fg_m.sqrt_g) * (da(u2_cov) - db(u1_cov))
            abs_vort = vort + fg_m.f_coriolis
            
            KE = 0.5 * (u1_cov * u1_con + u2_cov * u2_con)
            Phi = GRAVITY * h
            E = KE + Phi

            # 3. Fluxes 
            F_mass_1 = m * u1_con
            F_mass_2 = m * u2_con
            
            div_mass = da(F_mass_1) + db(F_mass_2)
            
            grad_E_1 = da(E)
            grad_E_2 = db(E)
            
            # Source Terms
            S_1 = fg_m.sqrt_g * u2_con * abs_vort
            S_2 = -fg_m.sqrt_g * u1_con * abs_vort
            
            # Volume Updates (Use .at[].set)
            # rhs[0, i] = ... -> rhs = rhs.at[0, i].set(...)
            rhs = rhs.at[0, i].set(-(div_mass))
            rhs = rhs.at[1, i].set(-grad_E_1 + S_1)
            rhs = rhs.at[2, i].set(-grad_E_2 + S_2)
            
            # 4. SAT / Numerical Flux
            for side in range(4):
                # Setup slicing logic (static unroll)
                if side == 0:   # West
                    idx_2d = (0, slice(None)) 
                    nx, ny = -1, 0
                    w_node = fg_m.walpha[0]
                    vn = -u1_con[idx_2d]
                    g1_b = fg_m.g1_vec[0, :]; g2_b = fg_m.g2_vec[0, :]
                    
                elif side == 1: # East
                    idx_2d = (-1, slice(None))
                    nx, ny = 1, 0
                    w_node = fg_m.walpha[-1]
                    vn = u1_con[idx_2d]
                    g1_b = fg_m.g1_vec[-1, :]; g2_b = fg_m.g2_vec[-1, :]
                    
                elif side == 2: # South
                    idx_2d = (slice(None), 0)
                    nx, ny = 0, -1
                    w_node = fg_m.wbeta[0]
                    vn = -u2_con[idx_2d]
                    g1_b = fg_m.g1_vec[:, 0]; g2_b = fg_m.g2_vec[:, 0]
                    
                elif side == 3: # North
                    idx_2d = (slice(None), -1)
                    nx, ny = 0, 1
                    w_node = fg_m.wbeta[-1]
                    vn = u2_con[idx_2d]
                    g1_b = fg_m.g1_vec[:, -1]; g2_b = fg_m.g2_vec[:, -1]
                
                sat_coeff = scale / w_node
                
                # 4.1 Boundary Data
                m_out, V_out = self._get_boundary_flux_data_jax(global_state, i, side, grid)
                m_in = m[idx_2d]
                
                # Project V_out
                u1_out_proj = V_out[:,0]*g1_b[:,0] + V_out[:,1]*g1_b[:,1] + V_out[:,2]*g1_b[:,2]
                u2_out_proj = V_out[:,0]*g2_b[:,0] + V_out[:,1]*g2_b[:,1] + V_out[:,2]*g2_b[:,2]
                
                # Reconstruct Contravariant
                g11 = jnp.sum(g1_b*g1_b, axis=1)
                g12 = jnp.sum(g1_b*g2_b, axis=1)
                g22 = jnp.sum(g2_b*g2_b, axis=1)
                det = g11*g22 - g12**2
                
                u1_con_out = (g22 * u1_out_proj - g12 * u2_out_proj) / det
                u2_con_out = (g11 * u2_out_proj - g12 * u1_out_proj) / det
                
                vn_out = u1_con_out * nx + u2_con_out * ny
                
                # 4.2 Wave Speed Correction
                h_in = m_in / fg_m.sqrt_g[idx_2d]
                h_out = m_out / fg_m.sqrt_g[idx_2d] 
                
                if side < 2:
                    g_ii = fg_m.g_inv[idx_2d][..., 0, 0]
                else:
                    g_ii = fg_m.g_inv[idx_2d][..., 1, 1]
                    
                c_in = jnp.sqrt(GRAVITY * h_in * g_ii)
                c_out = jnp.sqrt(GRAVITY * h_out * g_ii)
                
                wave_speed = jnp.maximum(jnp.abs(vn) + c_in, jnp.abs(vn_out) + c_out)
                
                # 4.3 SAT Mass
                sat_mass = sat_coeff * (0.5 * (m_out * vn_out - m_in * vn) - 0.5 * wave_speed * (m_out - m_in))
                rhs = rhs.at[0, i, idx_2d[0], idx_2d[1]].add(-sat_mass)
                
                # 4.4 SAT Momentum
                u1_in = u1_cov[idx_2d]; u2_in = u2_cov[idx_2d]
                
                KE_in = 0.5 * (u1_in * (u1_in*fg_m.g_inv[idx_2d][...,0,0] + u2_in*fg_m.g_inv[idx_2d][...,0,1]) + 
                               u2_in * (u1_in*fg_m.g_inv[idx_2d][...,1,0] + u2_in*fg_m.g_inv[idx_2d][...,1,1]))
                E_in_val = KE_in + GRAVITY * h_in
                
                KE_out = 0.5 * (u1_out_proj * u1_con_out + u2_out_proj * u2_con_out)
                E_out_val = KE_out + GRAVITY * h_out
                
                F_u1_in = E_in_val * nx; F_u1_out = E_out_val * nx 
                F_u2_in = E_in_val * ny; F_u2_out = E_out_val * ny
                
                sat_u1 = sat_coeff * (0.5 * (F_u1_out - F_u1_in) - 0.5 * wave_speed * (u1_out_proj - u1_in)) 
                sat_u2 = sat_coeff * (0.5 * (F_u2_out - F_u2_in) - 0.5 * wave_speed * (u2_out_proj - u2_in))
                
                rhs = rhs.at[1, i, idx_2d[0], idx_2d[1]].add(-sat_u1)
                rhs = rhs.at[2, i, idx_2d[0], idx_2d[1]].add(-sat_u2)
                
        return rhs

    def compute_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """Wrapper calling JAX JIT function."""
        # Convert inputs to JAX if needed (usually handled by JIT)
        return self._compute_rhs_core(t, state, self.grid_metrics, self.D)

    @partial(jit, static_argnums=(0,))
    def _step_core(self, t, state, dt, grid, D, rk_a, rk_b, filter_mat):
        """Pure JAX RK5 Step"""
        local_state = state
        du = jnp.zeros_like(state)
        
        # RK5
        for k in range(5):
            rhs = self._compute_rhs_core(t, local_state, grid, D)
            
            # du = a[k]*du + dt*rhs
            # Use jax.lax.cond for the 0.0 check or just arithmetic 
            # (arithmetic is fine, a[k]=0 handles it, but requires valid du)
            # Optimization:
            # val = rk_a[k] * du + dt * rhs
            # du = val
            du = rk_a[k] * du + dt * rhs
            local_state = local_state + rk_b[k] * du
            
        # Filter Momentum Only (Indices 1 and 2 from Vars dim 0)
        # Layout: (3, 6, N, N) -> slice (2, 6, N, N)
        mom = local_state[1:3, :, :, :] 
        mom = mom @ filter_mat.T # Beta dim
        mom = jnp.swapaxes(mom, -1, -2) 
        mom = mom @ filter_mat.T # Alpha dim
        mom = jnp.swapaxes(mom, -1, -2)
        
        local_state = local_state.at[1:3, :, :, :].set(mom)
        return local_state

    def step(self, t: float, state: Array, dt: float) -> Array:
        return self._step_core(t, state, dt, self.grid_metrics, self.D, 
                             self.rk_a, self.rk_b, self.filter_matrix)

    @partial(jit, static_argnums=(0, 9))
    def run_simulation_scan(self, state: Array, t_start: float, dt: float, grid, D, rk_a, rk_b, filter_mat, num_steps: int) -> Tuple[Array, float]:
        def scan_body(carry, _):
            s, t = carry
            s_new = self._step_core(t, s, dt, grid, D, rk_a, rk_b, filter_mat)
            return (s_new, t + dt), None

        (final_state, final_time), _ = lax.scan(scan_body, (state, t_start), None, length=num_steps)
        return final_state, final_time

    def solve(self, t_span: Tuple[float, float], initial_state: Union[np.ndarray, Array], callbacks: List[Any] = None) -> np.ndarray:
        """
        Solve the Shallow Water Equations over the given time span.

        Args:
            t_span: (t_start, t_end) tuple.
            initial_state: Initial state (NumPy or JAX array) of shape (3, 6, N+1, N+1).
            callbacks: Optional list of callables `cb(t, state_np)`.

        Returns:
            Final state as a NumPy array.

        Performance Notes:
            - **Fast Path (Recommended)**: If `callbacks` is None, the solver uses `jax.lax.scan`.
              This compiles the entire simulation into a single XLA kernel, providing massive
              speedups on accelerators.
            - **Slow Path (Debug)**: If `callbacks` are provided, the solver falls back to a 
              Python loop. This is useful for debugging or frequent I/O but suffers from 
              kernel launch overhead.
        """
        t, t_end = t_span
        # Ensure initial state is JAX array
        state = jnp.array(initial_state)
        
        dt_algo = self.config.get('dt', None)
        if dt_algo is None:
            dx_est = (self.R * 1.5) / self.N
            c_wave = math.sqrt(GRAVITY * self.config.get('H_avg', 10000.0))
            dt_algo = 0.5 * dx_est / c_wave 
            print(f"[JAX] Auto-estimated dt: {dt_algo:.4f}s")
        else:
            dt_algo = float(dt_algo)
            
        print(f"[JAX] Solving SWE: N={self.N}, dt={dt_algo:.4f}s, T_end={t_end}")
        
        # Initial Callback
        if callbacks:
            s_np = np.array(state)
            for cb in callbacks: cb(t, s_np)
            
        # Optimization: Use lax.scan if no sequential callbacks required
        if not callbacks:
            remaining_time = t_end - t
            num_steps = int(math.ceil(remaining_time / dt_algo))
            
            print(f"[JAX] Fast Path Activated: Compiling {num_steps} steps via lax.scan...")
            state, t = self.run_simulation_scan(state, t, dt_algo, self.grid_metrics, self.D, 
                                              self.rk_a, self.rk_b, self.filter_matrix, num_steps)
            
        else:
            print("[JAX] Slow Path: Callbacks detected, using Python loop.")
            step_count = 0
            epsilon = 1e-9
            while t < t_end - epsilon:
                remaining = t_end - t
                step_dt = min(dt_algo, remaining)
                
                state = self.step(t, state, step_dt)
                t += step_dt
                step_count += 1
                
                if callbacks:
                    s_np = np.array(state)
                    for cb in callbacks: cb(t, s_np)
                
        return np.array(state)

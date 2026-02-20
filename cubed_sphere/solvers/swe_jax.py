import jax
import jax.numpy as jnp
from jax import jit, lax, Array, vmap
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
    
    Architecture (Refactored for VMAP):
    - Metrics are stacked into shape (6, N, N) to allow parallel processing over faces.
    - Topology neighbor lookups are pre-computed into static index arrays.
    - Python loops over faces are replaced by `jax.vmap`.
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
        self.faces = {}
        for fname in self.topology.FACE_MAP:
            # Generate (NumPy)
            fg = self.geometry.generate_face(self.num_nodes, fname) 
            # Extend (NumPy -> JAX inside)
            self._compute_extended_metrics(fg, D_np) 
            self.faces[fname] = fg
            
        # 4. Freeze Grid for JAX (NumPy -> GlobalGrid PyTree via freeze_grid)
        self.grid_metrics = freeze_grid(self.faces)
        
        # 5. Prepare Stacked Metrics and Topology for VMAP
        self.stacked_metrics = self._stack_metrics(self.grid_metrics)
        self.neighbor_indices = self._compute_neighbor_indices()

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
        # Result is (3, 6, N, N)
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
        
        # Calculate Coriolis
        lam, theta = self.geometry.lonlat_from_xyz(fg.X, fg.Y, fg.Z)
        f_coriolis = 2.0 * OMEGA * np.sin(theta)
        
        # Explicit casts to self.dtype
        fg.g1_vec = jnp.array(g1_vec, dtype=self.dtype)
        fg.g2_vec = jnp.array(g2_vec, dtype=self.dtype)
        fg.g_inv = jnp.array(g_inv, dtype=self.dtype)
        fg.g_ij  = jnp.array(g_ij, dtype=self.dtype)
        fg.sqrt_g = jnp.array(fg.sqrt_g, dtype=self.dtype) # Use existing analytical sqrt_g
        fg.f_coriolis = jnp.array(f_coriolis, dtype=self.dtype)

        # Store lon/lat for utility (NumPy or JAX, doesn't matter for JIT as config)
        fg.lon = lam
        fg.lat = theta

    def _stack_metrics(self, global_grid: GlobalGrid) -> FaceMetrics:
        """
        Stack metrics from 6 faces into a single FaceMetrics object where each field
        has a leading dimension of 6.
        """
        # We assume FaceMetrics is a NamedTuple where all fields are Arrays
        fields = global_grid.faces[0]._fields
        stacked = {}
        for f in fields:
            # Stack [face0.field, face1.field, ...]
            # Must ensure order is correct P1..P6 (GlobalGrid already ensures this)
            stacked[f] = jnp.stack([getattr(face, f) for face in global_grid.faces], axis=0)
        
        return FaceMetrics(**stacked)

    def _compute_neighbor_indices(self) -> jnp.ndarray:
        """
        Builds a static index map for topology lookups.
        Shape: (6, 4, 3) 
        - Dim 0: Face Index (0..5)
        - Dim 1: Side Index (0..3) [West, East, South, North]
        - Dim 2: [Neighbor_Face_Idx, Neighbor_Side_Idx, Reverse_Flag(0/1)]
        """
        indices = np.zeros((6, 4, 3), dtype=np.int32)
        
        for face_idx in range(6):
            for side_idx in range(4):
                # CONN_TABLE maps (face, side) -> (nbr_face, nbr_side, swap_dim, reverse_dirs)
                nbr_face, nbr_side, swap, reverse = self.topology.CONN_TABLE[(face_idx, side_idx)]
                
                indices[face_idx, side_idx, 0] = nbr_face
                indices[face_idx, side_idx, 1] = nbr_side
                indices[face_idx, side_idx, 2] = 1 if reverse else 0
                
        return jnp.array(indices)


    # Remove static_argnums for self, as it may cause hashing issues if treated as Pytree or similar.
    # We pass all array args explicitly, so we don't need self to be static or dynamic really.
    # But since it's a method bound to self, JAX JIT on methods treats self as the first arg.
    # To be safe, we use 'static_argnums' only for integers/bools.
    @partial(jit, static_argnums=(0,))
    def _compute_rhs_core(self, t, global_state, stacked_metrics: FaceMetrics, D, neighbor_indices):
        """
        Fully vectorized RHS computation over faces.
        """
        # global_state: (3, 6, N, N) -> Transpose to (6, 3, N, N) for vmap
        dtype = global_state.dtype
        scale = jnp.array(4.0 / np.pi, dtype=dtype)
        state_T = jnp.swapaxes(global_state, 0, 1) # (6, 3, N, N)
        
        # --- 1. Volume Terms (Vectorized per face) ---
        
        def volume_kernel(state, metrics):
            # state: (3, N, N)
            m = state[0]
            u1_cov = state[1]
            u2_cov = state[2]
            
            def da(F): return D @ F * scale
            def db(F): return F @ D.T * scale
            
            h = m / metrics.sqrt_g
            
            # Contravariant
            g_inv = metrics.g_inv
            u1_con = g_inv[...,0,0]*u1_cov + g_inv[...,0,1]*u2_cov
            u2_con = g_inv[...,1,0]*u1_cov + g_inv[...,1,1]*u2_cov
            
            # Vorticity
            vort = (1.0 / metrics.sqrt_g) * (da(u2_cov) - db(u1_cov))
            abs_vort = vort + metrics.f_coriolis
            
            # Energy
            KE = 0.5 * (u1_cov * u1_con + u2_cov * u2_con)
            Phi = self.gravity * h
            E = KE + Phi
            
            # Fluxes
            F_mass_1 = m * u1_con
            F_mass_2 = m * u2_con
            
            div_mass = da(F_mass_1) + db(F_mass_2)
            
            grad_E_1 = da(E)
            grad_E_2 = db(E)
            
            S_1 = metrics.sqrt_g * u2_con * abs_vort
            S_2 = -metrics.sqrt_g * u1_con * abs_vort
            
            rhs_m = -(div_mass)
            rhs_u1 = -grad_E_1 + S_1
            rhs_u2 = -grad_E_2 + S_2
            
            return jnp.stack([rhs_m, rhs_u1, rhs_u2]), u1_con, u2_con
        
        # Perform Volume Op (6 copies in parallel)
        rhs_vol, u1_con_all, u2_con_all = vmap(volume_kernel)(state_T, stacked_metrics)
        
        # --- 2. Boundary / SAT Terms (Vectorized over Faces & Sides) ---
        
        # We loop over 4 sides (unrolled), but VMAP over faces for each side.
        rhs_final = rhs_vol # Initialize with volume terms
        
        # Pre-calculate helper values
        
        for side_idx in range(4):
            # --- Gather Logic ---
            # Indices for this side across all faces
            nbr_info = neighbor_indices[:, side_idx, :] # (6, 3)
            nbr_faces = nbr_info[:, 0]
            nbr_sides = nbr_info[:, 1]
            reverses  = nbr_info[:, 2] # (6,)
            
            # Helper: Extract edge slice from (6, 3, N, N) based on side index
            def get_edge_slice_vn(data_face, s_idx):
                # switch is safer than dynamic slice for XLA
                return lax.switch(s_idx, [
                    lambda x: x[:, 0, :],   # 0: West
                    lambda x: x[:, -1, :],  # 1: East
                    lambda x: x[:, :, 0],   # 2: South
                    lambda x: x[:, :, -1]   # 3: North
                ], data_face)

            # 1. Neighbor State
            nbr_states = state_T[nbr_faces] # Gather (6, 3, N, N)
            
            # Extract relevant edge from neighbor
            # vector vmap over (nbr_states, nbr_sides)
            nbr_edge_vals = vmap(get_edge_slice_vn)(nbr_states, nbr_sides) # (6, 3, N)

            # Apply Reverse if needed
            def apply_reverse(val, rev):
                # val (3, N), rev scalar 0/1
                return lax.cond(rev == 1, lambda v: v[:, ::-1], lambda v: v, val)
            nbr_edge_vals = vmap(apply_reverse)(nbr_edge_vals, reverses)

            # 2. Neighbor Metrics (Logic is complex: we need Neighbor V projected to OUR basis)
            # Instead of complex re-projection, we reconstruct V vector in 3D Cartesian
            # and then project it to *local* basis.
            
            # Gather NBR basis vectors
            nbr_g1 = stacked_metrics.g1_vec[nbr_faces] # (6, N, N, 3)
            nbr_g2 = stacked_metrics.g2_vec[nbr_faces]
            
            def get_edge_slice_vec(vec_face, s_idx):
                return lax.switch(s_idx, [
                    lambda x: x[0, :], 
                    lambda x: x[-1, :],
                    lambda x: x[:, 0],
                    lambda x: x[:, -1]
                ], vec_face)
            
            nbr_g1_edge = vmap(get_edge_slice_vec)(nbr_g1, nbr_sides) # (6, N, 3)
            nbr_g2_edge = vmap(get_edge_slice_vec)(nbr_g2, nbr_sides)
            
            # Apply reverse to vectors too
            def apply_reverse_vec(val, rev):
                return lax.cond(rev == 1, lambda v: v[::-1, :], lambda v: v, val)
            nbr_g1_edge = vmap(apply_reverse_vec)(nbr_g1_edge, reverses)
            nbr_g2_edge = vmap(apply_reverse_vec)(nbr_g2_edge, reverses)
            
            # Reconstruct Neighbor V in Cartesian (3D)
            u1_nb = nbr_edge_vals[:, 1, :]
            u2_nb = nbr_edge_vals[:, 2, :]
            
            # Calc Metric Tensor at NBR edge (to get Contravariant)
            ng11 = jnp.sum(nbr_g1_edge**2, axis=-1)
            ng12 = jnp.sum(nbr_g1_edge*nbr_g2_edge, axis=-1)
            ng22 = jnp.sum(nbr_g2_edge**2, axis=-1)
            ndet = ng11*ng22 - ng12**2
            
            u1_nb_con = (ng22 * u1_nb - ng12 * u2_nb) / ndet
            u2_nb_con = (ng11 * u2_nb - ng12 * u1_nb) / ndet
            
            # Neighbor Velocity Vector (3D Cartesian)
            V_nb_xyz = u1_nb_con[:, :, None] * nbr_g1_edge + u2_nb_con[:, :, None] * nbr_g2_edge
            
            # Mass out
            m_out = nbr_edge_vals[:, 0, :]
            
            # --- Local Context ---
            # We are at 'side_idx' (constant for all faces in this loop iteration)
            if side_idx == 0:
                # West Edge: varies in beta (Y), fixed alpha (X). Integration weight is wbeta properly?
                # Actually SAT penalty is (Flux - Flux) / weight_at_boundary_node.
                # If we are strictly 1D penalty, it is 1/w_i.
                # Here we are at alpha index 0. We need walpha[0].
                w_node = stacked_metrics.walpha[:, 0][:, None] # (6, 1)
                g1_local = stacked_metrics.g1_vec[:, 0, :]
                g2_local = stacked_metrics.g2_vec[:, 0, :]
                m_in = state_T[:, 0, 0, :]
                u1_in = state_T[:, 1, 0, :]
                u2_in = state_T[:, 2, 0, :]
                u_con_in = u1_con_all[:, 0, :]
                nx, ny = -1, 0
                idx_slice = (slice(None), 0, slice(None)) # For .at insert
            elif side_idx == 1:
                w_node = stacked_metrics.walpha[:, -1][:, None] # (6, 1)
                g1_local = stacked_metrics.g1_vec[:, -1, :]
                g2_local = stacked_metrics.g2_vec[:, -1, :]
                m_in = state_T[:, 0, -1, :]
                u1_in = state_T[:, 1, -1, :]
                u2_in = state_T[:, 2, -1, :]
                u_con_in = u1_con_all[:, -1, :]
                nx, ny = 1, 0
                idx_slice = (slice(None), -1, slice(None))
            elif side_idx == 2:
                w_node = stacked_metrics.wbeta[:, 0][:, None] # (6, 1)
                g1_local = stacked_metrics.g1_vec[:, :, 0]
                g2_local = stacked_metrics.g2_vec[:, :, 0]
                m_in = state_T[:, 0, :, 0]
                u1_in = state_T[:, 1, :, 0]
                u2_in = state_T[:, 2, :, 0]
                u_con_in = u2_con_all[:, :, 0]
                nx, ny = 0, -1
                idx_slice = (slice(None), slice(None), 0)
            elif side_idx == 3:
                w_node = stacked_metrics.wbeta[:, -1][:, None] # (6, 1)
                g1_local = stacked_metrics.g1_vec[:, :, -1]
                g2_local = stacked_metrics.g2_vec[:, :, -1]
                m_in = state_T[:, 0, :, -1]
                u1_in = state_T[:, 1, :, -1]
                u2_in = state_T[:, 2, :, -1]
                u_con_in = u2_con_all[:, :, -1]
                nx, ny = 0, 1
                idx_slice = (slice(None), slice(None), -1)

            # Project V_nb_xyz onto Local Basis (Covariant Components)
            u1_out_proj = jnp.sum(V_nb_xyz * g1_local, axis=-1)
            u2_out_proj = jnp.sum(V_nb_xyz * g2_local, axis=-1)
            
            # Convert Projected Covariant -> Contravariant (for Flux)
            lg11 = jnp.sum(g1_local**2, axis=-1)
            lg12 = jnp.sum(g1_local*g2_local, axis=-1)
            lg22 = jnp.sum(g2_local**2, axis=-1)
            ldet = lg11*lg22 - lg12**2
            
            u1_con_out = (lg22 * u1_out_proj - lg12 * u2_out_proj) / ldet
            u2_con_out = (lg11 * u2_out_proj - lg12 * u1_out_proj) / ldet
            
            vn_out = u1_con_out * nx + u2_con_out * ny
            vn_in = u_con_in * nx + u_con_in * ny * 0 # Just u_con_in * (1 or -1)
            # Actually u_con_in is already the component normal to boundary direction?
            # if side 0 (West), normal is -i. u_con_in is u1. vn = -u1.
            vn_in = lax.switch(side_idx, [
                lambda x: -x, lambda x: x, lambda x: -x, lambda x: x
            ], u_con_in)

            # --- Wave Speeds ---
             # h in/out
            # Access sqrt_g at edge
            sqrt_g_edge = lax.switch(side_idx, [
                lambda x: x[:, 0, :], lambda x: x[:, -1, :],
                lambda x: x[:, :, 0], lambda x: x[:, :, -1]
            ], stacked_metrics.sqrt_g)
            
            h_in = m_in / sqrt_g_edge
            h_out = m_out / sqrt_g_edge 
            
            # g_ii for c calculation (g^11 or g^22)
            if side_idx == 0:
                g_ii_edge = stacked_metrics.g_inv[..., 0, 0][:, 0, :]
            elif side_idx == 1:
                g_ii_edge = stacked_metrics.g_inv[..., 0, 0][:, -1, :]
            elif side_idx == 2:
                g_ii_edge = stacked_metrics.g_inv[..., 1, 1][:, :, 0]
            elif side_idx == 3:
                g_ii_edge = stacked_metrics.g_inv[..., 1, 1][:, :, -1]
                    
            c_in = jnp.sqrt(GRAVITY * h_in * g_ii_edge)
            c_out = jnp.sqrt(GRAVITY * h_out * g_ii_edge)
            wave_speed = jnp.maximum(jnp.abs(vn_in) + c_in, jnp.abs(vn_out) + c_out)
            
            sat_coeff = scale / w_node
            
            # --- SAT Terms ---
            sat_mass = sat_coeff * (0.5 * (m_out * vn_out - m_in * vn_in) - 0.5 * wave_speed * (m_out - m_in))
            
            # Momentum Fluxes (E * n)
            # Recompute Energy at edge
            # We need full u1_con, u2_con at edge for KE.
            # Local:
            def get_edge_slice_2d(data_face, s_idx):
                return lax.switch(s_idx, [
                    lambda x: x[0, :],   # 0: West (Row 0)
                    lambda x: x[-1, :],  # 1: East (Row -1)
                    lambda x: x[:, 0],   # 2: South (Col 0)
                    lambda x: x[:, -1]   # 3: North (Col -1)
                ], data_face)

            u1_con_in_full = vmap(get_edge_slice_2d)(u1_con_all, jnp.full((6,), side_idx, dtype=jnp.int32))
            u2_con_in_full = vmap(get_edge_slice_2d)(u2_con_all, jnp.full((6,), side_idx, dtype=jnp.int32))
            
            KE_in = 0.5 * (u1_in * u1_con_in_full + u2_in * u2_con_in_full)
            E_in_val = KE_in + GRAVITY * h_in
            
            KE_out = 0.5 * (u1_out_proj * u1_con_out + u2_out_proj * u2_con_out)
            E_out_val = KE_out + GRAVITY * h_out
            
            F_u1_in = E_in_val * nx; F_u1_out = E_out_val * nx
            F_u2_in = E_in_val * ny; F_u2_out = E_out_val * ny
            
            sat_u1 = sat_coeff * (0.5 * (F_u1_out - F_u1_in) - 0.5 * wave_speed * (u1_out_proj - u1_in))
            sat_u2 = sat_coeff * (0.5 * (F_u2_out - F_u2_in) - 0.5 * wave_speed * (u2_out_proj - u2_in))
            
            # --- Accumulate ---
            # We need to construct a mask/update.
            # We are updating rhs_final at 'idx_slice'.
            # Jax .at[].add() is standard.
            
            # Since side_idx is static loop, we construct specific updates.
            # Can we do this on (6, 3, N, N)? Yes.
            
            # Mass
            # Slice Logic (slice(None) for dims 0 and 2/3)
            # We use conditional add or specific indexing based on side_idx
            
            if side_idx == 0:
                rhs_final = rhs_final.at[:, 0, 0, :].add(-sat_mass)
                rhs_final = rhs_final.at[:, 1, 0, :].add(-sat_u1)
                rhs_final = rhs_final.at[:, 2, 0, :].add(-sat_u2)
            elif side_idx == 1:
                rhs_final = rhs_final.at[:, 0, -1, :].add(-sat_mass)
                rhs_final = rhs_final.at[:, 1, -1, :].add(-sat_u1)
                rhs_final = rhs_final.at[:, 2, -1, :].add(-sat_u2)
            elif side_idx == 2:
                rhs_final = rhs_final.at[:, 0, :, 0].add(-sat_mass)
                rhs_final = rhs_final.at[:, 1, :, 0].add(-sat_u1)
                rhs_final = rhs_final.at[:, 2, :, 0].add(-sat_u2)
            elif side_idx == 3:
                rhs_final = rhs_final.at[:, 0, :, -1].add(-sat_mass)
                rhs_final = rhs_final.at[:, 1, :, -1].add(-sat_u1)
                rhs_final = rhs_final.at[:, 2, :, -1].add(-sat_u2)

        return jnp.swapaxes(rhs_final, 0, 1) # Return (3, 6, N, N)

    def compute_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        return self._compute_rhs_core(t, state, self.stacked_metrics, self.D, self.neighbor_indices)

    @partial(jit, static_argnums=(0,))
    def _step_core(self, t, state, dt, stacked_metrics, D, rk_a, rk_b, filter_mat, neighbor_indices):
        """Pure JAX RK5 Step"""
        local_state = state
        du = jnp.zeros_like(state)
        
        # RK5
        for k in range(5):
            rhs = self._compute_rhs_core(t, local_state, stacked_metrics, D, neighbor_indices)
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
        return self._step_core(t, state, dt, self.stacked_metrics, self.D, 
                             self.rk_a, self.rk_b, self.filter_matrix, self.neighbor_indices)

    @partial(jit, static_argnums=(0, 10))
    def run_simulation_scan(self, state: Array, t_start: float, dt: float, stacked_metrics, D, rk_a, rk_b, filter_mat, neighbor_indices, num_steps: int) -> Tuple[Array, float]:
        def scan_body(carry, _):
            s, t = carry
            s_new = self._step_core(t, s, dt, stacked_metrics, D, rk_a, rk_b, filter_mat, neighbor_indices)
            return (s_new, t + dt), None

        (final_state, final_time), _ = lax.scan(scan_body, (state, t_start), None, length=num_steps)
        return final_state, final_time

    def _get_max_wave_speed(self, state):
        """
        JAX implementation of wave speed: v_max = max(|u| + sqrt(gH))
        """
        # 1. State Decomp (Assumed Valid JAX Arrays)
        # state: (3, 6, N, N)
        mass = state[0]      # h * sqrt_g
        u1_cov = state[1]    # Covariant u1
        u2_cov = state[2]    # Covariant u2
        
        # 2. Metrics (Stacked)
        metrics = self.stacked_metrics
        sqrt_g = metrics.sqrt_g
        g_inv = metrics.g_inv # (6, N, N, 2, 2)
        
        # Height h = mass / sqrt_g
        h = mass / sqrt_g
        
        # Contravariant Velocity (u^i = g^ij u_j)
        u1_con = g_inv[...,0,0]*u1_cov + g_inv[...,0,1]*u2_cov
        u2_con = g_inv[...,1,0]*u1_cov + g_inv[...,1,1]*u2_cov
        
        # Velocity Magnitude Squared (|u|^2 = u_i u^i)
        u_sq = u1_cov * u1_con + u2_cov * u2_con
        u_mag = jnp.sqrt(u_sq)
        
        # Gravity Wave Speed
        c_wave = jnp.sqrt(self.gravity * h)
        
        # Max characteristic speed (Global Max)
        lambda_max = jnp.max(u_mag + c_wave)
        return float(lambda_max) # Ensure Python float for dt calc

    def solve(self, t_span: Tuple[float, float], initial_state: Union[np.ndarray, Array], callbacks: List[Any] = None) -> np.ndarray:
        t, t_end = t_span
        state = jnp.array(initial_state)
        
        dt_algo = self.config.get('dt', None)
        if dt_algo is None:
            # Auto-compute CFL-based dt
            cfl = self.config.get('CFL', 0.1)
            dt_algo = self.compute_safe_dt(state, cfl=cfl)
            print(f"[JAX] Auto-computed Safe dt: {dt_algo:.6f}s (CFL={cfl})")
        else:
            dt_algo = float(dt_algo)
            
        print(f"[JAX] Solving SWE: N={self.N}, dt={dt_algo:.4f}s, T_end={t_end}")
        
        if callbacks:
            s_np = np.array(state)
            for cb in callbacks: cb(t, s_np)
            
        if not callbacks:
            remaining_time = t_end - t
            num_steps = int(math.ceil(remaining_time / dt_algo))
            
            print(f"[JAX] Fast Path Activated: Compiling {num_steps} steps via lax.scan...")
            state, t = self.run_simulation_scan(state, t, dt_algo, self.stacked_metrics, self.D, 
                                              self.rk_a, self.rk_b, self.filter_matrix, self.neighbor_indices, num_steps)
            
        else:
            # Slow path logic... (omitted for brevity, largely identical to before but using step())
            remaining_time = t_end - t
            num_steps = int(math.ceil(remaining_time / dt_algo))
            for i in range(num_steps):
                state = self.step(t, state, dt_algo)
                t += dt_algo
                s_np = np.array(state)
                for cb in callbacks: cb(t, s_np)

        return np.array(state)

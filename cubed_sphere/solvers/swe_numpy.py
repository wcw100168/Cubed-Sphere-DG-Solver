import numpy as np
import dataclasses
from typing import Dict, Tuple, List, Any, Optional
from cubed_sphere.solvers.base import BaseSolver
from cubed_sphere.geometry.grid import CubedSphereTopology, CubedSphereEquiangular, FaceGrid
from cubed_sphere.numerics.spectral import lgl_diff_matrix, lgl_nodes_weights
from cubed_sphere.physics.initialization import get_initial_state
from numpy.polynomial.legendre import Legendre
import math

# Physics Constants
EARTH_RADIUS = 6.37122e6
OMEGA = 7.292e-5
GRAVITY = 9.80616

def compute_bv_filter_matrix(N: int, strength=36.0, order=16) -> np.ndarray:
    """
    Computes the Boyd-Vandeven filter matrix F = V * Lambda * V^-1.
    
    Args:
        N: Polynomial order (Number of nodes will be N+1).
        strength: Filter strength parameter (alpha).
        order: Filter order (s).
        
    Returns:
        F: Filter matrix of shape (N+1, N+1).
    """
    # 1. Get LGL nodes (physical grid points)
    # The spectral.py lgl_nodes_weights takes 'number of nodes'.
    # If N is poly order, nodes = N + 1.
    num_nodes = N + 1
    nodes, _ = lgl_nodes_weights(-1.0, 1.0, num_nodes)
    
    # 2. Construct Vandermonde Matrix V
    # V[i, j] = P_j(x_i)
    V = np.zeros((num_nodes, num_nodes))
    for j in range(num_nodes):
        Pj = Legendre.basis(j)
        V[:, j] = Pj(nodes)
        
    # 3. Construct Diagonal Spectral dampening matrix Lambda
    # sigma_k = exp( -strength * (k/N)**order )
    sigma = np.zeros(num_nodes)
    for k in range(num_nodes):
        if N > 0:
            sigma[k] = np.exp(-strength * (float(k)/N)**order)
        else:
            sigma[k] = 1.0
        
    Lambda = np.diag(sigma)
    
    # 4. Compute F = V @ Lambda @ V_inv
    # Use pseudo-inverse for stability, though square matrix usually fine
    V_inv = np.linalg.inv(V)
    F = V @ Lambda @ V_inv
    
    return F

@dataclasses.dataclass
class SWEStateConfig:
    """Configuration for the SWE Solver state."""
    N: int
    dt: float = 0.0
    
class CubedSphereSWENumpy(BaseSolver):
    """
    Shallow Water Equations solver on Cubed Sphere using NumPy.
    Implements the Vector Invariant Formulation with Rusanov Fluxes.
    Follows a stateless design pattern.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the solver geometry and operators.
        
        Args:
            config: Dictionary containing 'N' (polynomial order), 'R' (radius), etc.
        """
        super().__init__(config)
        self.N = config.get('N', 32)
        self.R = config.get('R', EARTH_RADIUS)
        
        # 1. Topology & Geometry
        self.topology = CubedSphereTopology()
        self.D = lgl_diff_matrix(self.N + 1) # N is poly order, diff matrix needs N+1 nodes? 
        # CAUTION: Check numerics/spectral.py lgl_diff_matrix definition.
        # Usually lgl_diff_matrix(N) creates NxN matrix for N nodes.
        # If config['N'] is Poly Order, nodes = N+1.
        # But previous swe.py used config.N for matrix size directly.
        # The notebook explicitly calls `lgl_diff_matrix(N)` where N was passed as 16 (Poly Order?).
        # Wait, Notebook `CubedSphereSWE(N=16)` -> `geometry = ...(N, self.D)`.
        # Notebook `FaceGrid` uses `lgl_nodes_weights(..., self.N)`.
        # So N in notebook is Number of Nodes?
        # Notebook output "Step ... Max H...". The prompt says `(6, 3, N+1, N+1)` for state shape.
        # This implies N is Poly Order.
        # Standard convention: N = Polynomial Degree. Grid points = N+1.
        # The package `swe.py` uses `lgl_diff_matrix(config.N)`.
        # I will assume config['N'] is Polynomial Order, so I need N+1 nodes.
        self.num_nodes = self.N + 1
        self.D = lgl_diff_matrix(self.num_nodes)
        
        self.geometry = CubedSphereEquiangular(self.R)
        
        # 2. Operators
        # Filter Matrix (Order 16, Strength 36.0)
        self.filter_matrix = compute_bv_filter_matrix(self.N, strength=36.0, order=16)
        
        # RK5 Coefficients
        self.rk_a = np.array([0.0, 
                              -567301805773.0/1357537059087.0, 
                              -2404267990393.0/2016746695238.0, 
                              -3550918686646.0/2091501179385.0, 
                              -1275806237668.0/842570457699.0])
        self.rk_b = np.array([1432997174477.0/9575080441755.0, 
                              5161836677717.0/13612068292357.0, 
                              1720146321549.0/2090206949498.0, 
                              3134564353537.0/4481467310338.0, 
                              2277821191437.0/14882151754819.0])
        
        # 3. Initialize Faces and Extended Metrics
        self.faces = {}
        for fname in self.topology.FACE_MAP:
            # Base generation from package
            fg = self.geometry.generate_face(self.num_nodes, fname) 
            # Note: generate_face in package takes Num_Nodes
            
            # Compute Extended Metrics (g_ij, g_inv, Coriolis) needed for this solver
            self._compute_extended_metrics(fg)
            self.faces[fname] = fg

    def get_initial_condition(self, type: str = None, **kwargs) -> np.ndarray:
        """
        Generate initial conditions (state variables).
        Delegates to shared physics module.
        """
        if type is None:
            type = self.config.get('initial_condition', "case2")
            
        if isinstance(type, int):
            type = f"case{type}"
            
        return get_initial_state(self.config, self.faces, type, **kwargs)

    def _compute_extended_metrics(self, fg: FaceGrid):
        """
        Augment the FaceGrid object with strict metric tensor components
        required by the Vector Invariant Formulation.
        """
        # Re-compute X, Y, Z derivatives locally using D matrix
        # Note: mapping is on [-1, 1] for D.
        # generate_face maps nodes to Equiangular, which implies a range.
        # But D is spectral Differentiation on the index space.
        # We need the mapping factor if the D matrix is on [-1, 1] 
        # but the physical variable is parameterized by alpha in [-pi/4, pi/4].
        # d/dAlpha = (d/dZeta) * (dZeta/dAlpha).
        # Zeta [-1, 1], Alpha [-pi/4, pi/4] -> Scale = 2 / (pi/2) = 4/pi.
        scale = 4.0 / np.pi
        
        # We can apply D to the Cartesian coordinates X, Y, Z (which are computed in generate_face)
        # to get basis vectors (tangent vectors).
        
        # g1 = d/dAlpha, g2 = d/dBeta
        g1_x = self.D @ fg.X * scale
        g2_x = fg.X @ self.D.T * scale # Note: Right multiplication for Beta derivative
        g1_y = self.D @ fg.Y * scale
        g2_y = fg.Y @ self.D.T * scale
        g1_z = self.D @ fg.Z * scale
        g2_z = fg.Z @ self.D.T * scale
        
        # Pack vectors: (N, N, 3)
        fg.g1_vec = np.stack([g1_x, g1_y, g1_z], axis=-1)
        fg.g2_vec = np.stack([g2_x, g2_y, g2_z], axis=-1)
        
        # Metric Tensor g_ij (Covariant)
        g11 = np.sum(fg.g1_vec * fg.g1_vec, axis=-1)
        g22 = np.sum(fg.g2_vec * fg.g2_vec, axis=-1)
        g12 = np.sum(fg.g1_vec * fg.g2_vec, axis=-1)
        
        # Inverse Metric g^ij (Contravariant)
        det = g11*g22 - g12**2
        inv_det = 1.0 / det
        
        # fg.sqrt_g = np.sqrt(det) # Numerical Jacobian (optional, currently using Analytic)
        
        # Store as (N, N, 2, 2)
        fg.g_ij = np.zeros(g11.shape + (2, 2))
        fg.g_ij[..., 0, 0] = g11
        fg.g_ij[..., 1, 1] = g22
        fg.g_ij[..., 0, 1] = g12
        fg.g_ij[..., 1, 0] = g12
        
        fg.g_inv = np.zeros(g11.shape + (2, 2))
        fg.g_inv[..., 0, 0] = g22 * inv_det
        fg.g_inv[..., 1, 1] = g11 * inv_det
        fg.g_inv[..., 0, 1] = -g12 * inv_det
        fg.g_inv[..., 1, 0] = -g12 * inv_det
        
        # Jacobian (Override/Verify existing sqrt_g)
        # fg.sqrt_g = np.sqrt(det) # Use Analytical Jacobian from Geometry instead of Numerical
        
        # Coriolis
        lam, theta = self.geometry.lonlat_from_xyz(fg.X, fg.Y, fg.Z)
        fg.f_coriolis = 2.0 * OMEGA * np.sin(theta)
        
        # Store lon/lat for utility
        fg.lon = lam
        fg.lat = theta

    def _get_boundary_flux_data(self, global_state, face_idx, side):
        """Helper to get neighbor state for Rusanov flux"""
        nbr_face, nbr_side, swap, reverse = self.topology.CONN_TABLE[(face_idx, side)]
        nbr_name = self.topology.FACE_MAP[nbr_face]
        nbr_fg = self.faces[nbr_name]
        
        # Correctly access state for neighbor face in Var-Major layout (Vars, Face, Xi, Eta)
        # global_state shape is (3, 6, N, N)
        nbr_U = global_state[:, nbr_face, :, :] # (3, N, N)
        
        # Convention: Dim 1 = Alpha (X), Dim 2 = Beta (Y)
        # Note: state shape is (3, N, N). Dim 1 is Alpha?
        # Typically (Var, Alpha, Beta).
        
        if nbr_side == 0:   slc = (slice(None), 0, slice(None))  # Alpha=0
        elif nbr_side == 1: slc = (slice(None), -1, slice(None)) # Alpha=-1
        elif nbr_side == 2: slc = (slice(None), slice(None), 0)  # Beta=0
        elif nbr_side == 3: slc = (slice(None), slice(None), -1) # Beta=-1
            
        # Extract Data
        # nbr_U is (3, N, N).
        # We want to slice the last two dims.
        # Let's rebuild slice for (3, N, N)
        if nbr_side == 0:   s = (slice(None), 0, slice(None))
        elif nbr_side == 1: s = (slice(None), -1, slice(None))
        elif nbr_side == 2: s = (slice(None), slice(None), 0)
        elif nbr_side == 3: s = (slice(None), slice(None), -1)
        
        val = nbr_U[s] # (3, N)
        m_nb = val[0]
        u1_nb = val[1]
        u2_nb = val[2]
        
        if reverse:
            m_nb = m_nb[::-1]; u1_nb = u1_nb[::-1]; u2_nb = u2_nb[::-1]
            
        # Project Vectors
        if nbr_side == 0:   g1, g2 = nbr_fg.g1_vec[0, :], nbr_fg.g2_vec[0, :]
        elif nbr_side == 1: g1, g2 = nbr_fg.g1_vec[-1, :], nbr_fg.g2_vec[-1, :]
        elif nbr_side == 2: g1, g2 = nbr_fg.g1_vec[:, 0], nbr_fg.g2_vec[:, 0]
        elif nbr_side == 3: g1, g2 = nbr_fg.g1_vec[:, -1], nbr_fg.g2_vec[:, -1]
        
        if reverse:
            g1 = g1[::-1]; g2 = g2[::-1]

        # Reconstruct Cartesian V
        g11 = np.sum(g1*g1, axis=1); g12 = np.sum(g1*g2, axis=1); g22 = np.sum(g2*g2, axis=1)
        det = g11*g22 - g12**2
        
        u1_contra = (g22 * u1_nb - g12 * u2_nb) / det
        u2_contra = (g11 * u2_nb - g12 * u1_nb) / det
        
        Vx = u1_contra * g1[:,0] + u2_contra * g2[:,0]
        Vy = u1_contra * g1[:,1] + u2_contra * g2[:,1]
        Vz = u1_contra * g1[:,2] + u2_contra * g2[:,2]
        
        return m_nb, np.stack([Vx, Vy, Vz], axis=-1)

    def compute_rhs(self, t: float, global_state: np.ndarray) -> np.ndarray:
        """
        Compute RHS of the Shallow Water Equations (vector-invariant form).

        global_state shape:
            (3, 6, N+1, N+1)  -> [variable, face, alpha, beta]

        Variables:
            0 : m      = h * sqrt(g)
            1 : u1_cov = covariant velocity component u_α
            2 : u2_cov = covariant velocity component u_β
        """
        rhs = np.zeros_like(global_state)
        scale = 4.0 / np.pi

        # Spectral derivatives in reference space
        def da(F): return self.D @ F * scale
        def db(F): return F @ self.D.T * scale

        # ------------------------------------------------------------------
        # Loop over cubed-sphere faces
        # ------------------------------------------------------------------
        for i, fname in enumerate(self.topology.FACE_MAP):

            fg = self.faces[fname]

            # --- Unpack state ------------------------------------------------
            m       = global_state[0, i]
            u1_cov  = global_state[1, i]
            u2_cov  = global_state[2, i]

            h = m / fg.sqrt_g

            # ==============================================================
            # 1. Contravariant velocity: u^i = g^{ij} u_j
            # ==============================================================
            u1_con = fg.g_inv[..., 0, 0] * u1_cov + fg.g_inv[..., 0, 1] * u2_cov
            u2_con = fg.g_inv[..., 1, 0] * u1_cov + fg.g_inv[..., 1, 1] * u2_cov

            # ==============================================================
            # 2. Vorticity, energy, and Bernoulli function
            # ==============================================================
            vort = (da(u2_cov) - db(u1_cov)) / fg.sqrt_g
            abs_vort = vort + fg.f_coriolis

            KE  = 0.5 * (u1_cov * u1_con + u2_cov * u2_con)
            Phi = GRAVITY * h
            E   = KE + Phi

            # ==============================================================
            # 3. Interior fluxes
            # ==============================================================

            # --- Mass conservation ----------------------------------------
            F_mass_1 = m * u1_con
            F_mass_2 = m * u2_con
            div_mass = da(F_mass_1) + db(F_mass_2)

            # --- Momentum equation (vector-invariant form) ----------------
            grad_E_1 = da(E)
            grad_E_2 = db(E)

            S_1 =  fg.sqrt_g * u2_con * abs_vort
            S_2 = -fg.sqrt_g * u1_con * abs_vort

            rhs[0, i] = -div_mass
            rhs[1, i] = -grad_E_1 + S_1
            rhs[2, i] = -grad_E_2 + S_2

            # ==============================================================
            # 4. SAT boundary fluxes
            # ==============================================================

            for side in range(4):

                # ----------------------------------------------------------
                # Boundary geometry & normal direction
                # ----------------------------------------------------------
                if side == 0:      # West (α = 0)
                    idx = (0, slice(None))
                    nx, ny = -1, 0
                    w_node = fg.walpha[0]
                    vn = -u1_con[idx]
                    g1_b = fg.g1_vec[0, :]
                    g2_b = fg.g2_vec[0, :]

                elif side == 1:    # East (α = N)
                    idx = (-1, slice(None))
                    nx, ny = 1, 0
                    w_node = fg.walpha[-1]
                    vn = u1_con[idx]
                    g1_b = fg.g1_vec[-1, :]
                    g2_b = fg.g2_vec[-1, :]

                elif side == 2:    # South (β = 0)
                    idx = (slice(None), 0)
                    nx, ny = 0, -1
                    w_node = fg.wbeta[0]
                    vn = -u2_con[idx]
                    g1_b = fg.g1_vec[:, 0]
                    g2_b = fg.g2_vec[:, 0]

                else:              # North (β = N)
                    idx = (slice(None), -1)
                    nx, ny = 0, 1
                    w_node = fg.wbeta[-1]
                    vn = u2_con[idx]
                    g1_b = fg.g1_vec[:, -1]
                    g2_b = fg.g2_vec[:, -1]

                sat_coeff = scale / w_node

                # ----------------------------------------------------------
                # 4.1 External boundary state
                # ----------------------------------------------------------
                m_out, V_out = self._get_boundary_flux_data(global_state, i, side)
                m_in = m[idx]

                # Project external Cartesian velocity onto local covariant basis
                u1_out = np.sum(V_out * g1_b, axis=1)
                u2_out = np.sum(V_out * g2_b, axis=1)

                # Reconstruct contravariant components
                g11 = np.sum(g1_b * g1_b, axis=1)
                g12 = np.sum(g1_b * g2_b, axis=1)
                g22 = np.sum(g2_b * g2_b, axis=1)
                det = g11 * g22 - g12**2

                u1_con_out = (g22 * u1_out - g12 * u2_out) / det
                u2_con_out = (g11 * u2_out - g12 * u1_out) / det

                vn_out = u1_con_out * nx + u2_con_out * ny

                # ----------------------------------------------------------
                # 4.2 Wave speed (Rusanov type)
                # ----------------------------------------------------------
                h_in  = m_in  / fg.sqrt_g[idx]
                h_out = m_out / fg.sqrt_g[idx]

                g_ii = fg.g_inv[idx][..., 0, 0] if side < 2 \
                    else fg.g_inv[idx][..., 1, 1]

                c_in  = np.sqrt(GRAVITY * h_in  * g_ii)
                c_out = np.sqrt(GRAVITY * h_out * g_ii)

                wave_speed = np.maximum(np.abs(vn) + c_in,
                                        np.abs(vn_out) + c_out)

                # ----------------------------------------------------------
                # 4.3 SAT mass flux
                # ----------------------------------------------------------
                sat_mass = sat_coeff * (
                    0.5 * (m_out * vn_out - m_in * vn)
                    - 0.5 * wave_speed * (m_out - m_in)
                )

                rhs[0, i][idx] -= sat_mass

                # ----------------------------------------------------------
                # 4.4 SAT momentum flux (energy form)
                # ----------------------------------------------------------
                u1_in = u1_cov[idx]
                u2_in = u2_cov[idx]

                g_inv_loc = fg.g_inv[idx]

                KE_in = 0.5 * (
                    u1_in * (u1_in * g_inv_loc[..., 0, 0] +
                            u2_in * g_inv_loc[..., 0, 1]) +
                    u2_in * (u1_in * g_inv_loc[..., 1, 0] +
                            u2_in * g_inv_loc[..., 1, 1])
                )

                E_in  = KE_in + GRAVITY * h_in
                KE_out = 0.5 * (u1_out * u1_con_out +
                                u2_out * u2_con_out)
                E_out = KE_out + GRAVITY * h_out

                F_u1_in  = E_in  * nx
                F_u1_out = E_out * nx
                F_u2_in  = E_in  * ny
                F_u2_out = E_out * ny

                sat_u1 = sat_coeff * (
                    0.5 * (F_u1_out - F_u1_in)
                    - 0.5 * wave_speed * (u1_out - u1_in)
                )

                sat_u2 = sat_coeff * (
                    0.5 * (F_u2_out - F_u2_in)
                    - 0.5 * wave_speed * (u2_out - u2_in)
                )

                rhs[1, i][idx] -= sat_u1
                rhs[2, i][idx] -= sat_u2

        return rhs

    def step(self, t: float, state: np.ndarray, dt: float) -> np.ndarray:
        """
        Perform one time step using LSRK5 + Filter.
        """
        local_state = state.copy()
        du = np.zeros_like(state)
        
        # 1. RK5 Stage
        for k in range(5):
            rhs = self.compute_rhs(t, local_state) # t is approx? Vector invariant is usually autonomous
            
            # du = a[k]*du + dt*rhs
            if self.rk_a[k] == 0.0:
                 du = dt * rhs
            else:
                 du = self.rk_a[k] * du + dt * rhs
            
            # u = u + b[k]*du
            local_state += self.rk_b[k] * du
            
        # 2. Filter (Momentum Only)
        # state is (3, 6, N, N) [Var-Major]. We filter indices 1 and 2.
        # Tensor product: F @ U @ F.T apply along last two dims
        # F is (Nodes, Nodes).
        
        # Slice momentum: (2, 6, N, N)
        mom = local_state[1:3, :, :, :]
        
        # Apply F on last axis (Dimension Beta)
        # mom @ F.T
        mom = mom @ self.filter_matrix.T
        
        # Apply F on second to last axis (Dimension Alpha)
        # We want F @ mom_slice. 
        # Swap axes to put Alpha last: (2, 6, Beta, Alpha)
        mom = np.swapaxes(mom, -1, -2) 
        # Apply F to Alpha
        mom = mom @ self.filter_matrix.T
        # Swap back
        mom = np.swapaxes(mom, -1, -2)
        
        # Update state
        local_state[1:3, :, :, :] = mom
        
        return local_state

    def _get_max_wave_speed(self, state):
        """
        NumPy implementation of V_max = max(|u| + sqrt(gH))
        state: (3, 6, N, N)
        """
        # Mass: (6, N, N)
        mass = state[0]
        
        # Need sqrt_g for each face.
        # solver.faces is dict, map to array (6, N, N)
        # Assuming P1..P6 order matches state index 0..5
        # We can construct these on the fly or cache them.
        # Since this is NumPy and init time, we loop once.
        
        lambda_max = 0.0
        
        face_names = self.topology.FACE_MAP
        for i, fname in enumerate(face_names):
            if fname not in self.faces: continue
            
            fg = self.faces[fname]
            m = mass[i]
            h = m / fg.sqrt_g
            
            # Covariant
            u1_cov = state[1, i]
            u2_cov = state[2, i]
            
            # Metric Tensor
            g_inv = fg.g_inv # (N, N, 2, 2)
            
            # Contravariant
            u1_con = g_inv[...,0,0]*u1_cov + g_inv[...,0,1]*u2_cov
            u2_con = g_inv[...,1,0]*u1_cov + g_inv[...,1,1]*u2_cov
            
            # Mag Squared
            u_sq = u1_cov * u1_con + u2_cov * u2_con
            u_mag = np.sqrt(np.maximum(u_sq, 0)) # Clip negative zeros
            
            c_wave = np.sqrt(GRAVITY * h)
            
            face_max = np.max(u_mag + c_wave)
            lambda_max = max(lambda_max, face_max)
            
        return float(lambda_max)

    def solve(self, t_span: Tuple[float, float], initial_state: np.ndarray, callbacks: List[Any] = None) -> np.ndarray:
        """
        Run simulation from t_span[0] to t_span[1] starting with initial_state.
        """
        t, t_end = t_span
        state = initial_state.copy()
        dt_algo = self.config.get('dt', None)
        
        # Automatic dt estimation if not provided
        if dt_algo is None:
            cfl = self.config.get('CFL', 0.1)
            dt_algo = self.compute_safe_dt(state, cfl=cfl)
            print(f"[NumPy] Auto-computed Safe dt: {dt_algo:.6f}s (CFL={cfl})")
        else:
            dt_algo = float(dt_algo)
            
        print(f"Solving SWE (NumPy): N={self.N}, dt={dt_algo:.4f}s, T={t_end}")
        
        step_count = 0
        epsilon = 1e-9
        
        if callbacks:
            for cb in callbacks: cb(t, state)
            
        while t < t_end - epsilon:
            remaining = t_end - t
            step_dt = min(dt_algo, remaining)
            
            state = self.step(t, state, step_dt)
            t += step_dt
            step_count += 1
            
            if callbacks:
                for cb in callbacks: cb(t, state)
                
        return state

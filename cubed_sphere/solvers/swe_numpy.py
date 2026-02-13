import numpy as np
import dataclasses
from typing import Dict, Tuple, List, Any, Optional
from cubed_sphere.solvers.base import BaseSolver
from cubed_sphere.geometry.grid import CubedSphereTopology, CubedSphereEquiangular, FaceGrid
from cubed_sphere.numerics.spectral import lgl_diff_matrix, lgl_nodes_weights
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

    def get_initial_condition(self, type: str = "case2", **kwargs) -> np.ndarray:
        """
        Generate initial conditions on the solver's grid.
        Leverages the internally computed metrics (sqrt_g) for geometric consistency.
        
        Args:
            type: "case2" (Williamson 2), "flat" (Static H=H_avg), etc.
        """
        state = np.zeros((3, 6, self.num_nodes, self.num_nodes))
        
        # Parameters
        R = self.R
        Omega = OMEGA
        g = GRAVITY
        u0_vel = 2.0 * np.pi * R / (12.0 * 24.0 * 3600.0) # ~38 m/s
        h0 = self.config.get('H_avg', 10000.0)
        
        if type == "flat":
             for i, fname in enumerate(self.topology.FACE_MAP):
                fg = self.faces[fname]
                # H = h0 -> Mass = h0 * sqrt_g
                state[0, i] = h0 * fg.sqrt_g
                state[1, i] = 0.0
                state[2, i] = 0.0
             return state
             
        elif type == "case2":
            # Williamson Case 2: Steady State Zonal Flow
            alpha_deg = kwargs.get('alpha', 0.0) # Angle of rotation (degrees)
            alpha = np.deg2rad(alpha_deg)
            sin_alpha = np.sin(alpha)
            cos_alpha = np.cos(alpha)
            
            for i, fname in enumerate(self.topology.FACE_MAP):
                fg = self.faces[fname]
                
                # Coordinates
                lam = fg.lon
                th = fg.lat
                
                # 1. Rotated Latitude sin(theta') (Eq 2.4 Williamson)
                # sin(th') = sin(th)cos(a) - cos(lam)cos(th)sin(a)
                sin_th_prime = np.sin(th) * cos_alpha - np.cos(lam) * np.cos(th) * sin_alpha
                
                # 2. Height Field (Eq 2.3 Williamson)
                # h = h0 - (1/g)(R*Om*u0 + u0^2/2) * sin^2(th')
                term = (R * Omega * u0_vel + 0.5 * u0_vel**2)
                h = h0 - (term * sin_th_prime**2) / g
                
                # Initial Mass = h * sqrt_g
                state[0, i] = h * fg.sqrt_g
                
                # 3. Velocity Field (Eq 2.1 & 2.2 Williamson)
                # u_s = u0 * (cos(th)cos(a) + cos(lam)sin(th)sin(a))
                # v_s = -u0 * sin(lam)sin(a)
                u_sph = u0_vel * (np.cos(th) * cos_alpha + np.cos(lam) * np.sin(th) * sin_alpha)
                v_sph = -u0_vel * np.sin(lam) * sin_alpha
                
                # Project onto Local Contravariant Basis
                # We need Covariant components (u_1, u_2) for the state.
                # Project (u_sph, v_sph) -> Cartesian V -> Dot with covariant basis vectors g_1, g_2
                # V = u_s * e_lam + v_s * e_th
                
                sin_lam, cos_lam = np.sin(fg.lon), np.cos(fg.lon)
                sin_th, cos_th = np.sin(fg.lat), np.cos(fg.lat)
                
                e_lam_x, e_lam_y, e_lam_z = -sin_lam, cos_lam, 0.0
                e_th_x, e_th_y, e_th_z = -sin_th*cos_lam, -sin_th*sin_lam, cos_th
                
                Vx = u_sph * e_lam_x + v_sph * e_th_x
                Vy = u_sph * e_lam_y + v_sph * e_th_y
                Vz = u_sph * e_lam_z + v_sph * e_th_z
                
                # Covariant components u_i = V . g_i
                # g_i vectors are stored in fg.g1_vec, fg.g2_vec
                
                b1 = Vx*fg.g1_vec[...,0] + Vy*fg.g1_vec[...,1] + Vz*fg.g1_vec[...,2]
                b2 = Vx*fg.g2_vec[...,0] + Vy*fg.g2_vec[...,1] + Vz*fg.g2_vec[...,2]
                
                state[1, i] = b1
                state[2, i] = b2
            return state

        elif type == "case6":
            # Williamson Case 6: Rossby-Haurwitz Wave
            # Parameters
            omega = kwargs.get('omega', 7.848e-6)
            K = kwargs.get('K', 7.848e-6)
            R_wave = kwargs.get('R_wave', 4.0)
            h0_c6 = kwargs.get('h0', 8000.0)
            
            # Constants
            R_earth = self.config['R'] # Radius
            Omega = self.config['Omega']
            g = self.config['gravity']
            
            for i, fname in enumerate(self.topology.FACE_MAP):
                fg = self.faces[fname]
                
                # Coordinates
                lam = fg.lon
                th = fg.lat
                
                sin_lat = np.sin(th)
                cos_lat = np.cos(th)
                cos_Rlam = np.cos(R_wave * lam)
                sin_Rlam = np.sin(R_wave * lam)
                cos_2Rlam = np.cos(2.0 * R_wave * lam)
                
                # Height Field Functions (A, B, C)
                # A(theta)
                # term1 = (w/2)*(2*Om + w)*cos^2(th)
                # term2 = (1/4)*K^2 * cos^(2R)(th) * [...]
                # Ref: Williamson (1992) Eq 4.6
                t1 = (omega / 2.0) * (2.0 * Omega + omega) * (cos_lat**2)
                
                # Numerically safe computation for t2 to avoid pole singularity cos^(-2)
                # Expand cos^(2R) into the bracket terms:
                # [ (R+1)cos^2 + (2R^2 - R - 2) - 2R^2 cos^(-2) ] * cos^(2R)
                # = (R+1)cos^(2R+2) + (2R^2 - R - 2)cos^(2R) - 2R^2 cos^(2R-2)
                
                c2R = cos_lat**(2*R_wave)
                c2R_plus2 = cos_lat**(2*R_wave + 2)
                c2R_minus2 = cos_lat**(2*R_wave - 2)
                
                t2 = 0.25 * K**2 * (
                    (R_wave + 1) * c2R_plus2 + 
                    (2 * R_wave**2 - R_wave - 2) * c2R - 
                    2 * R_wave**2 * c2R_minus2
                )
                A = t1 + t2
                
                # B(theta) - Eq 4.7
                # term = (2*(Om+w)*K) / ((R+1)(R+2)) * cos^R(th) * [...]
                b_num = 2.0 * (Omega + omega) * K
                b_den = (R_wave + 1) * (R_wave + 2)
                b_brack = ( (R_wave**2 + 2*R_wave + 2) - (R_wave+1)**2 * cos_lat**2 )
                B = (b_num / b_den) * (cos_lat**R_wave) * b_brack
                
                # C(theta) - Eq 4.8
                # term = (1/4)*K^2 * cos^(2R)(th) * [ (R+1)cos^2(th) - (R+2) ]
                c_brack = (R_wave + 1) * cos_lat**2 - (R_wave + 2)
                C = 0.25 * K**2 * c2R * c_brack
                
                # Total Height h
                # gh = gh0 + a^2 A + a^2 B cos(R lam) + a^2 C cos(2R lam)
                gh = g * h0_c6 + (R_earth**2) * (A + B * cos_Rlam + C * cos_2Rlam)
                h = gh / g
                
                # Store Mass: h * sqrt_g
                state[0, i] = h * fg.sqrt_g
                
                # Velocity Field
                # u_sph = a*w*cos(th) + a*K*cos^(R-1)(th) * (R*sin^2(th) - cos^2(th)) * cos(R*lam)
                # v_sph = -a*K*R * cos^(R-1)(th) * sin(th) * sin(R*lam)
                
                u_term1 = R_earth * omega * cos_lat
                u_term2 = R_earth * K * (cos_lat**(R_wave-1)) * (R_wave * sin_lat**2 - cos_lat**2) * cos_Rlam
                u_sph = u_term1 + u_term2
                
                v_sph = -R_earth * K * R_wave * (cos_lat**(R_wave-1)) * sin_lat * sin_Rlam
                
                # Project to Covariant
                # Convert to Cartesian V
                sin_lam, cos_lam = np.sin(lam), np.cos(lam)
                # e_lam = (-sin(lon), cos(lon), 0)
                # e_th = (-sin(lat)cos(lon), -sin(lat)sin(lon), cos(lat))
                
                Vx = u_sph * (-sin_lam) + v_sph * (-sin_lat * cos_lam)
                Vy = u_sph * (cos_lam)  + v_sph * (-sin_lat * sin_lam)
                Vz = u_sph * (0.0)      + v_sph * (cos_lat)
                
                # Covariant u_1, u_2 = V . g_1, V . g_2
                b1 = Vx*fg.g1_vec[...,0] + Vy*fg.g1_vec[...,1] + Vz*fg.g1_vec[...,2]
                b2 = Vx*fg.g2_vec[...,0] + Vy*fg.g2_vec[...,1] + Vz*fg.g2_vec[...,2]
                
                state[1, i] = b1
                state[2, i] = b2

            return state
        
        else:
            raise NotImplementedError(f"Condition {type} not implemented")

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
        Compute du/dt for the Shallow Water Equations.
        Global State: (3, 6, N+1, N+1)  [Var-Major]
        """
        rhs = np.zeros_like(global_state)
        scale = 4.0 / np.pi
        
        # d/dAlpha, d/dBeta
        def da(F): return self.D @ F * scale
        def db(F): return F @ self.D.T * scale
        
        for i, fname in enumerate(self.topology.FACE_MAP):
            fg = self.faces[fname]
            
            # Unpack Variables (3, N, N) from Var-Major state
            m = global_state[0, i]
            u1_cov = global_state[1, i]
            u2_cov = global_state[2, i]
            
            h = m / fg.sqrt_g
            
            # 1. Contravariant Velocities
            # u^i = g^ij u_j
            u1_con = fg.g_inv[...,0,0]*u1_cov + fg.g_inv[...,0,1]*u2_cov
            u2_con = fg.g_inv[...,1,0]*u1_cov + fg.g_inv[...,1,1]*u2_cov
            
            # 2. Vorticity & Energy
            vort = (1.0 / fg.sqrt_g) * (da(u2_cov) - db(u1_cov))
            abs_vort = vort + fg.f_coriolis
            
            KE = 0.5 * (u1_cov * u1_con + u2_cov * u2_con)
            Phi = GRAVITY * h
            E = KE + Phi

            # 3. Fluxes 
            F_mass_1 = m * u1_con
            F_mass_2 = m * u2_con
            
            div_mass = da(F_mass_1) + db(F_mass_2)
            
            grad_E_1 = da(E)
            grad_E_2 = db(E)
            
            # Source Terms (Vector Invariant Cor/Vort Force)
            S_1 = fg.sqrt_g * u2_con * abs_vort
            S_2 = -fg.sqrt_g * u1_con * abs_vort
            
            rhs[0, i] = -(div_mass)
            rhs[1, i] = -grad_E_1 + S_1
            rhs[2, i] = -grad_E_2 + S_2
            
            # 4. SAT / Numerical Flux (Boundaries)
            for side in range(4):
                if side == 0:   # West
                    idx = (slice(None), 0, slice(None)) # (3, N, N) -> last 2 dims are space
                    # Oops, idx must index into (N,N). 
                    # If side=0 (Alpha=0), we slice Dim 0 of the N,N block?
                    # wait, standard layout is (Alpha, Beta).
                    # Side 0 is Alpha Min (0).
                    # Indexing for RHS (3, N, N):
                    idx_rhs = (slice(None), 0, slice(None)) # Variable, Alpha=0, Beta=:
                    idx_2d = (0, slice(None)) # Alpha=0, Beta=:
                    
                    nx, ny = -1, 0
                    w_node = fg.walpha[0]
                    vn = -u1_con[idx_2d]
                    
                    # Boundary vectors
                    g1_b = fg.g1_vec[0, :]; g2_b = fg.g2_vec[0, :]
                    
                elif side == 1: # East
                    idx_rhs = (slice(None), -1, slice(None))
                    idx_2d = (-1, slice(None))
                    nx, ny = 1, 0
                    w_node = fg.walpha[-1]
                    vn = u1_con[idx_2d]
                    g1_b = fg.g1_vec[-1, :]; g2_b = fg.g2_vec[-1, :]
                    
                elif side == 2: # South
                    idx_rhs = (slice(None), slice(None), 0)
                    idx_2d = (slice(None), 0)
                    nx, ny = 0, -1
                    w_node = fg.wbeta[0]
                    vn = -u2_con[idx_2d]
                    g1_b = fg.g1_vec[:, 0]; g2_b = fg.g2_vec[:, 0]
                    
                elif side == 3: # North
                    idx_rhs = (slice(None), slice(None), -1)
                    idx_2d = (slice(None), -1)
                    nx, ny = 0, 1
                    w_node = fg.wbeta[-1]
                    vn = u2_con[idx_2d]
                    g1_b = fg.g1_vec[:, -1]; g2_b = fg.g2_vec[:, -1]
                
                # SAT Weight
                sat_coeff = scale / w_node
                
                # 4.1 Get External Data
                m_out, V_out = self._get_boundary_flux_data(global_state, i, side)
                m_in = m[idx_2d]
                
                # Project V_out to local vectors
                # V_out is (N, 3). g1_b is (N, 3).
                u1_out_proj = V_out[:,0]*g1_b[:,0] + V_out[:,1]*g1_b[:,1] + V_out[:,2]*g1_b[:,2]
                u2_out_proj = V_out[:,0]*g2_b[:,0] + V_out[:,1]*g2_b[:,1] + V_out[:,2]*g2_b[:,2]
                
                # Reconstruct Contravariant
                g11 = np.sum(g1_b*g1_b, axis=1)
                g12 = np.sum(g1_b*g2_b, axis=1)
                g22 = np.sum(g2_b*g2_b, axis=1)
                det = g11*g22 - g12**2
                
                u1_con_out = (g22 * u1_out_proj - g12 * u2_out_proj) / det
                u2_con_out = (g11 * u2_out_proj - g12 * u1_out_proj) / det
                
                vn_out = u1_con_out * nx + u2_con_out * ny
                
                # 4.2 Wave Speed Correction
                h_in = m_in / fg.sqrt_g[idx_2d]
                h_out = m_out / fg.sqrt_g[idx_2d] # Assuming conformal match of sqrt_g at boundary
                
                # Metric term g^ii
                if side < 2:
                    g_ii = fg.g_inv[idx_2d][..., 0, 0]
                else:
                    g_ii = fg.g_inv[idx_2d][..., 1, 1]
                    
                c_in = np.sqrt(GRAVITY * h_in * g_ii)
                c_out = np.sqrt(GRAVITY * h_out * g_ii)
                
                wave_speed = np.maximum(np.abs(vn) + c_in, np.abs(vn_out) + c_out)
                
                # 4.3 SAT Mass
                sat_mass = sat_coeff * (0.5 * (m_out * vn_out - m_in * vn) - 0.5 * wave_speed * (m_out - m_in))
                rhs[0, i][idx_2d] -= sat_mass
                
                # 4.4 SAT Momentum (Energy Flux)
                u1_in = u1_cov[idx_2d]; u2_in = u2_cov[idx_2d]
                
                # Kinetic Energy In
                # We need full contraction g^ij * u_i * u_j
                # It's explicitly written in Notebook
                KE_in = 0.5 * (u1_in * (u1_in*fg.g_inv[idx_2d][...,0,0] + u2_in*fg.g_inv[idx_2d][...,0,1]) + 
                               u2_in * (u1_in*fg.g_inv[idx_2d][...,1,0] + u2_in*fg.g_inv[idx_2d][...,1,1]))
                E_in_val = KE_in + GRAVITY * h_in
                
                KE_out = 0.5 * (u1_out_proj * u1_con_out + u2_out_proj * u2_con_out)
                E_out_val = KE_out + GRAVITY * h_out
                
                F_u1_in = E_in_val * nx; F_u1_out = E_out_val * nx 
                F_u2_in = E_in_val * ny; F_u2_out = E_out_val * ny
                
                sat_u1 = sat_coeff * (0.5 * (F_u1_out - F_u1_in) - 0.5 * wave_speed * (u1_out_proj - u1_in)) 
                sat_u2 = sat_coeff * (0.5 * (F_u2_out - F_u2_in) - 0.5 * wave_speed * (u2_out_proj - u2_in))
                
                rhs[1, i][idx_2d] -= sat_u1
                rhs[2, i][idx_2d] -= sat_u2
                
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

    def solve(self, t_span: Tuple[float, float], initial_state: np.ndarray, callbacks: List[Any] = None) -> np.ndarray:
        """
        Run simulation from t_span[0] to t_span[1] starting with initial_state.
        """
        t, t_end = t_span
        state = initial_state.copy()
        dt_algo = self.config.get('dt', None)
        
        # Automatic dt estimation if not provided
        if dt_algo is None:
            # Estimate C = sqrt(gH) ~ 300 m/s
            # dx ~ R / N. (Actually pi R / 2 / N)
            dx_est = (self.R * 1.5) / self.N
            c_wave = math.sqrt(GRAVITY * self.config.get('H_avg', 10000.0))
            dt_algo = 0.5 * dx_est / c_wave # CFL ~ 0.5
            print(f"Auto-estimated dt: {dt_algo:.4f}s")
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

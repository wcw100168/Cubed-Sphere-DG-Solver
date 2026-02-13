import numpy as np
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Dict, Optional, List
import math
import dataclasses
from cubed_sphere.numerics.spectral import lgl_nodes_weights

class FaceMetrics(NamedTuple):
    """Immutable container for static geometric data of a single face."""
    # Metric Terms
    sqrt_g: jnp.ndarray      # Jacobian (N+1, N+1)
    g_ij: jnp.ndarray        # Covariant Metric Tensor (N+1, N+1, 2, 2)
    g_inv: jnp.ndarray       # Contravariant Metric Tensor (N+1, N+1, 2, 2)
    
    # 3D Basis Vectors (Critical for Flux Projection)
    g1_vec: jnp.ndarray      # Tangent vector 1 in 3D cartesian (N+1, N+1, 3)
    g2_vec: jnp.ndarray      # Tangent vector 2 in 3D cartesian (N+1, N+1, 3)
    
    # Coriolis
    f_coriolis: jnp.ndarray  # Coriolis parameter (N+1, N+1)
    
    # Quadrature Weights
    walpha: jnp.ndarray      # (N+1,)
    wbeta: jnp.ndarray       # (N+1,)

class GlobalGrid(NamedTuple):
    """Container for the entire spherical grid."""
    faces: Tuple[FaceMetrics, ...]  # Tuple containing metrics for 6 faces

@dataclasses.dataclass
class FaceGrid:
    """
    Grid data and precomputed metric terms for a single face of the cubed sphere.
    """
    name: str
    alpha: np.ndarray
    beta:  np.ndarray
    walpha: np.ndarray # Weights
    wbeta: np.ndarray
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray      # 3D Cart X
    Y: np.ndarray      # 3D Cart Y
    Z: np.ndarray      # 3D Cart Z
    
    # --- Pre-computed Metric Terms ---
    sqrt_g: Optional[np.ndarray] = None  # Jacobian
    u1: Optional[np.ndarray] = None      # Contravariant velocity 1 (static wind)
    u2: Optional[np.ndarray] = None      # Contravariant velocity 2 (static wind)
    div_u: Optional[np.ndarray] = None   # Divergence of static wind field
    
    # --- Dynamic State ---
    phi: Optional[np.ndarray] = None     # Scalar field

class CubedSphereTopology:
    """
    Handles connectivity between faces of the Cubed-Sphere.
    """
    FACE_MAP: List[str] = ["P1", "P2", "P3", "P4", "P5", "P6"]
    FACE_IDX: Dict[str, int] = {name: i for i, name in enumerate(FACE_MAP)}
    
    # Connection Table: (FaceIdx, Side) -> (NeighborFaceIdx, NeighborSide, SwapXY, Reverse)
    # Sides: 0:West(Alpha=-1), 1:East(Alpha=1), 2:South(Beta=-1), 3:North(Beta=1)
    CONN_TABLE: Dict[Tuple[int, int], Tuple[int, int, bool, bool]] = {
        (0, 0): (3, 1, False, False), (0, 1): (1, 0, False, False), (0, 2): (5, 3, False, False), (0, 3): (4, 2, False, False),
        (1, 0): (0, 1, False, False), (1, 1): (2, 0, False, False), (1, 2): (5, 1, True, True),   (1, 3): (4, 1, True, False),
        (2, 0): (1, 1, False, False), (2, 1): (3, 0, False, False), (2, 2): (5, 2, False, True),  (2, 3): (4, 3, False, True),
        (3, 0): (2, 1, False, False), (3, 1): (0, 0, False, False), (3, 2): (5, 0, True, False),  (3, 3): (4, 0, True, True),
        (4, 0): (3, 3, True, True),   (4, 1): (1, 3, True, False),  (4, 2): (0, 3, False, False), (4, 3): (2, 3, False, True),
        (5, 0): (3, 2, True, False),  (5, 1): (1, 2, True, True),   (5, 2): (2, 2, False, True),  (5, 3): (0, 2, False, False),
    }

    def get_neighbor_data(self, global_state: np.ndarray, face_idx: int, side_idx: int) -> np.ndarray:
        """
        Extract neighbor boundary data from global state, handling flips and rotations.
        
        Parameters
        ----------
        global_state : np.ndarray
            Shape (6, N, N) containing data for all faces.
        face_idx : int
            Index of the current face.
        side_idx : int
            Side index (0:West, 1:East, 2:South, 3:North).
            
        Returns
        -------
        slice_data : np.ndarray
            1D array of neighbor boundary data.
        """
        nbr_face, nbr_side, swap, reverse = self.CONN_TABLE[(face_idx, side_idx)]
        
        # Access neighbor face data
        # global_state form: (..., 6, N, N) or (6, N, N)
        # We assume the second to last dimension is Face Index if it matches 6?
        # Actually standard is (n_vars, 6, N, N) or (6, N, N).
        # We index the dimension corresponding to faces. 
        # But 'global_state' could be varying shapes.
        # Let's assume standard layout where face dim is -3 (N,N are -1,-2).
        
        nbr_data = global_state[..., nbr_face, :, :]
        
        # Extract slice
        if nbr_side == 0: 
            slice_data = nbr_data[..., 0, :]   # West edge
        elif nbr_side == 1: 
            slice_data = nbr_data[..., -1, :]  # East edge
        elif nbr_side == 2: 
            slice_data = nbr_data[..., :, 0]   # South edge
        elif nbr_side == 3: 
            slice_data = nbr_data[..., :, -1]  # North edge
        else:
            raise ValueError(f"Invalid side index: {nbr_side}")
        
        if reverse:
            slice_data = slice_data[..., ::-1]
            
        return slice_data

def freeze_grid(faces: Dict[str, FaceGrid]) -> GlobalGrid:
    """
    Convert legacy dictionary of FaceGrids into an immutable GlobalGrid.
    """
    metrics_list = []
    # Ensure correct order P1...P6
    face_names = ["P1", "P2", "P3", "P4", "P5", "P6"]
    
    for fname in face_names:
        fg = faces[fname]
        
        # Helper to safely convert numpy -> jax.numpy
        def to_jax(arr):
            # If it's already jax array?
            return jnp.array(arr)
            
        metrics = FaceMetrics(
            sqrt_g=to_jax(fg.sqrt_g),
            g_ij=to_jax(fg.g_ij),
            g_inv=to_jax(fg.g_inv),
            g1_vec=to_jax(fg.g1_vec),
            g2_vec=to_jax(fg.g2_vec),
            f_coriolis=to_jax(fg.f_coriolis),
            walpha=to_jax(fg.walpha),
            wbeta=to_jax(fg.wbeta)
        )
        metrics_list.append(metrics)
        
    return GlobalGrid(faces=tuple(metrics_list))

class CubedSphereEquiangular:
    """
    Geometry transformations and metric tensor calculations for the Equiangular Cubed Sphere.
    """
    def __init__(self, R: float = 1.0):
        self.R = float(R)
        self.a = self.R / math.sqrt(3.0)

    def generate_face(self, N: int, face: str) -> FaceGrid:
        """
        Generate grid and coordinates for a single face.
        """
        # LGL Nodes [-pi/4, pi/4]
        nodes, weights = lgl_nodes_weights(-np.pi/4, np.pi/4, N)
        AA, BB = np.meshgrid(nodes, nodes, indexing="ij")
        
        # Equiangular coordinates to local x, y
        x = self.a * np.tan(AA)
        y = self.a * np.tan(BB)
        
        # Local x, y to Global X, Y, Z
        X, Y, Z = self._xyz_from_xy(x, y, face)
        
        fg = FaceGrid(name=face, alpha=AA, beta=BB, walpha=weights, wbeta=weights,
                      x=x, y=y, X=X, Y=Y, Z=Z)
        
        # Calculate sqrt_g (Jacobian) immediately
        rho = np.sqrt(1.0 + np.tan(AA)**2 + np.tan(BB)**2)
        fg.sqrt_g = (self.R**2) / (rho**3 * (np.cos(AA)**2) * (np.cos(BB)**2))
        
        return fg

    def _xyz_from_xy(self, x: np.ndarray, y: np.ndarray, face: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Map local face coordinates (x, y) to global Cartesian (X, Y, Z).
        """
        r = np.sqrt(self.a**2 + x**2 + y**2)
        s = self.R / r
        
        if face == "P1":   return s*self.a, s*x, s*y
        elif face == "P2": return s*(-x), s*self.a, s*y
        elif face == "P3": return s*(-self.a), s*(-x), s*y
        elif face == "P4": return s*x, s*(-self.a), s*y
        elif face == "P5": return s*(-y), s*x, s*self.a
        elif face == "P6": return s*y, s*x, s*(-self.a)
        else: raise ValueError(f"Unknown face: {face}")

    @staticmethod
    def lonlat_from_xyz(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert Cartesian (X, Y, Z) to Spherical (Lambda, Theta).
        """
        lam = np.arctan2(Y, X)
        theta = np.arcsin(Z / np.sqrt(X**2 + Y**2 + Z**2))
        return lam, theta

    def solid_body_wind(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, alpha0: float, u0: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Solid Body Rotation wind field (u, v) in Spherical components.
        """
        lam, theta = self.lonlat_from_xyz(X, Y, Z)
        u_sphere = u0 * (np.cos(alpha0) * np.cos(theta) + np.sin(alpha0) * np.cos(lam) * np.sin(theta))
        v_sphere = -u0 * np.sin(alpha0) * np.sin(lam)
        return u_sphere, v_sphere

    def compute_contravariant_vel(self, fg: FaceGrid, u_sphere: np.ndarray, v_sphere: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute contravariant velocities u1, u2 from spherical velocities.
        
        Uses the mapping from Sadourny/Nair.
        """
        lam, theta = self.lonlat_from_xyz(fg.X, fg.Y, fg.Z)
        
        # A matrix components construction
        sec2_a = 1.0 / (np.cos(fg.alpha) ** 2)
        sec2_b = 1.0 / (np.cos(fg.beta) ** 2)
        c_lam, s_lam = np.cos(lam), np.sin(lam)
        c_th, s_th = np.cos(theta), np.sin(theta)
        
        A = np.zeros(fg.alpha.shape + (2, 2))
        
        # Populate A matrix based on face
        if fg.name == "P1":
            A[...,0,0], A[...,0,1] = self.R * c_lam**2 * c_th * sec2_a, 0.0
            A[...,1,0], A[...,1,1] = -self.R * s_lam * c_lam * s_th * c_th * sec2_a, self.R * c_lam * c_th**2 * sec2_b
        elif fg.name == "P2":
            A[...,0,0], A[...,0,1] = self.R * s_lam**2 * c_th * sec2_a, 0.0
            A[...,1,0], A[...,1,1] = self.R * s_lam * c_lam * s_th * c_th * sec2_a, self.R * s_lam * c_th**2 * sec2_b
        elif fg.name == "P3":
            A[...,0,0], A[...,0,1] = self.R * c_lam**2 * c_th * sec2_a, 0.0
            A[...,1,0], A[...,1,1] = -self.R * s_lam * c_lam * s_th * c_th * sec2_a, -self.R * c_lam * c_th**2 * sec2_b
        elif fg.name == "P4":
            A[...,0,0], A[...,0,1] = self.R * s_lam**2 * c_th * sec2_a, 0.0
            A[...,1,0], A[...,1,1] = self.R * s_lam * c_lam * s_th * c_th * sec2_a, -self.R * s_lam * c_th**2 * sec2_b
        elif fg.name == "P5": # Top
            A[...,0,0], A[...,0,1] = self.R * c_lam * s_th * sec2_a, self.R * s_lam * s_th * sec2_b
            A[...,1,0], A[...,1,1] = -self.R * s_lam * s_th**2 * sec2_a, self.R * c_lam * s_th**2 * sec2_b
        elif fg.name == "P6": # Bottom
            A[...,0,0], A[...,0,1] = -self.R * c_lam * s_th * sec2_a, self.R * s_lam * s_th * sec2_b
            A[...,1,0], A[...,1,1] = self.R * s_lam * s_th**2 * sec2_a, self.R * c_lam * s_th**2 * sec2_b

        # Inversion A^-1 * [u, v]^T
        det = A[...,0,0]*A[...,1,1] - A[...,0,1]*A[...,1,0]
        inv_det = 1.0 / det
        
        u1 = inv_det * (A[...,1,1]*u_sphere - A[...,0,1]*v_sphere)
        u2 = inv_det * (-A[...,1,0]*u_sphere + A[...,0,0]*v_sphere)
        return u1, u2

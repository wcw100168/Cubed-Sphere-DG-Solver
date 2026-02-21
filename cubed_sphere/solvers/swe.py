import numpy as np
import dataclasses
from typing import Dict, Tuple, List, Any, Optional
from cubed_sphere.solvers.base import BaseSolver
from cubed_sphere.solvers.advection import AdvectionConfig

@dataclasses.dataclass
class SWEConfig(AdvectionConfig):
    """
    Configuration for the Shallow Water Equations Solver.
    """
    H_avg: float = 10000.0   # Mean depth (m). For Case 2, can use 29400/g ~ 2998m
    Omega: float = 7.292e-5  # Rotation rate (rad/s)
    gravity: float = 9.80616 # Gravity (m/s^2)
    n_vars: int = 3          # h, u1, u2
    filter_order: int = 16   # Boyd-Vandeven filter order (16 is good for N>=16)
    dt: Optional[float] = None # Time step (s)
    
class CubedSphereSWE(BaseSolver):
    """
    Facade for the Shallow Water Equations Solver.
    Delegates to a backend-specific implementation (NumPy or JAX).
    
    This solver implements the full Non-Linear Shallow Water Equations 
    in Vector Invariant Form with Rusanov Fluxes.
    """
    
    def __init__(self, config: SWEConfig):
        # BaseSolver expects a dictionary for config to use .get()
        super().__init__(dataclasses.asdict(config))
        # We also store the strongly-typed config for internal usage
        self.swe_config = config
        self.N = config.N  # Required for BaseSolver.compute_safe_dt

        
        # Dispatch Backend
        if config.backend == 'numpy':
            from cubed_sphere.solvers.swe_numpy import CubedSphereSWENumpy as SWENumpySolver
            self._impl = SWENumpySolver(dataclasses.asdict(config))
        elif config.backend == 'jax':
             from cubed_sphere.solvers.swe_jax import CubedSphereSWEJax as SWEJaxSolver
             self._impl = SWEJaxSolver(dataclasses.asdict(config))
        else:
            raise ValueError(f"Unknown backend: {config.backend}")

    def _to_jax(self):
        """Delegates JAX conversion to implementation."""
        if hasattr(self._impl, '_to_jax'):
            self._impl._to_jax()
            
    def compute_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute the Right-Hand Side (du/dt).
        """
        return self._impl.compute_rhs(t, state)

    def _get_max_wave_speed(self, state: np.ndarray) -> float:
        """
        Delegates max wave speed calculation to the backend implementation.
        """
        if hasattr(self._impl, '_get_max_wave_speed'):
            return self._impl._get_max_wave_speed(state)
        else:
            raise NotImplementedError("Backend implementation missing '_get_max_wave_speed' method")

    def step(self, t: float, state: np.ndarray, dt: float) -> np.ndarray:
        """
        Perform a single time step (RK5 + Filter).
        """
        if hasattr(self._impl, 'step'):
            return self._impl.step(t, state, dt)
        else:
            raise NotImplementedError("Backend implementation missing 'step' method")

    def solve(self, t_span: Tuple[float, float], initial_state: np.ndarray, callbacks: List[Any] = None) -> np.ndarray:
        """
        Run the simulation.
        """
        return self._impl.solve(t_span, initial_state, callbacks)
        
    def get_initial_condition(self, type: str = "case2", **kwargs) -> np.ndarray:
        """
        Helper to generate initial conditions using the implementation's geometry.
        """
        # Delegate to implementation if it supports it, otherwise fallback
        if hasattr(self._impl, 'get_initial_condition'):
             return self._impl.get_initial_condition(type=type, **kwargs)
             
        if type == "case2":
            return self._init_case2()
        else:
            raise NotImplementedError(f"Initial condition type '{type}' not implemented for SWE.")

    def _init_case2(self) -> np.ndarray:
        """
        Sets up Williamson Case 2 (Global Steady State Zonal Flow).
        Returns the initial state array (3, 6, N+1, N+1).
        """
        config = self.swe_config
        num_nodes = config.N + 1
        state = np.zeros((3, 6, num_nodes, num_nodes))
        
        # Physics Parameters
        u0 = 2.0 * np.pi * config.R / (12.0 * 24.0 * 3600.0) # ~38 m/s
        h0 = config.H_avg # Base depth
        g = config.gravity
        R = config.R
        Omega = config.Omega
        
        topo = self._impl.topology
        faces = self._impl.faces
        
        for i, fname in enumerate(topo.FACE_MAP):
            fg = faces[fname]
            
            # Get coordinates (NumPy extraction if needed)
            # Assuming backend faces expose numpy-compatible properties or we cast
            if hasattr(fg.lat, 'device') or hasattr(fg.lat, 'sharding'):
                lat = np.array(fg.lat); lon = np.array(fg.lon)
                g1_vec = np.array(fg.g1_vec); g2_vec = np.array(fg.g2_vec)
                sqrt_g = np.array(fg.sqrt_g)
            else:
                lat = fg.lat; lon = fg.lon
                g1_vec = fg.g1_vec; g2_vec = fg.g2_vec
                sqrt_g = fg.sqrt_g
                
            # Analytical Solution (Spherical)
            u_sph = u0 * np.cos(lat)
            # v_sph = 0.0
            h_val = h0 - (R * Omega * u0 + 0.5 * u0**2) * (np.sin(lat)**2) / g
            
            # Basis Vectors
            e_lam_x = -np.sin(lon)
            e_lam_y = np.cos(lon)
            # e_lam_z = 0
            
            # Velocity Vector V (only zonal comp)
            Vx = u_sph * e_lam_x
            Vy = u_sph * e_lam_y
            Vz = np.zeros_like(Vx)
            
            # Project onto Covariant Basis (u_1, u_2)
            V_dot_g1 = Vx * g1_vec[..., 0] + Vy * g1_vec[..., 1] + Vz * g1_vec[..., 2]
            V_dot_g2 = Vx * g2_vec[..., 0] + Vy * g2_vec[..., 1] + Vz * g2_vec[..., 2]
            
            state[0, i] = h_val * sqrt_g
            state[1, i] = V_dot_g1
            state[2, i] = V_dot_g2
            
        return state

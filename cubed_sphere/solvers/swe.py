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
        super().__init__(dataclasses.asdict(config))
        self.config = config
        
        # Dispatch Backend
        if config.backend == 'numpy':
            from cubed_sphere.solvers.swe_numpy import CubedSphereSWENumpy as SWENumpySolver
            self._impl = SWENumpySolver(dataclasses.asdict(config))
        elif config.backend == 'jax':
             from cubed_sphere.solvers.swe_jax import CubedSphereSWEJax as SWEJaxSolver
             self._impl = SWEJaxSolver(dataclasses.asdict(config))
        else:
            raise ValueError(f"Unknown backend: {config.backend}")
            
    def compute_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute the Right-Hand Side (du/dt).
        """
        return self._impl.compute_rhs(t, state)

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
        Wrapper around manual setup or delegates if implementation supports it.
        """
        # Ideally move initial condition logic to a shared utility or the implementation
        # For now, we assume user sets up IC externally or we add helper here later.
        pass

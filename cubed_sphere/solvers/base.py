from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple, List

class BaseSolver(ABC):
    """
    Abstract base class for Cubed Sphere PDE solvers.
    Defines the physics and solver mechanics, independent of the state.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def compute_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute the right-hand side (RHS) of the PDE: d(state)/dt = RHS."""
        pass

    @abstractmethod
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
        pass
        
    @abstractmethod
    def solve(self, t_span: Tuple[float, float], initial_state: np.ndarray, callbacks: List[Any] = None) -> np.ndarray:
        """
        Run the simulation over t_span = (t_start, t_end) starting from initial_state.
        Returns the final state.
        
        callbacks: List of callables, called as cb(t, state) at each step.
        """
        pass

    def compute_safe_dt(self, state, cfl: float = 0.1) -> float:
        """
        Computes a stable time step (dt) using the Discontinuous Galerkin (DG) CFL condition.
        
        Formula: dt = CFL * (2.0 / (2.0 * V_max * (N ** 2))) 
        Where:
            V_max: Maximum characteristic wave speed in the domain.
            N: Polynomial order of the basis functions.
        
        Args:
            state: Current state vector (numpy or jax array).
            cfl: Courant number (default 0.1 for high-order DG).
            
        Returns:
            dt (float): The computed time step in seconds.
        """
        # 1. Calculate Global Max Wave Speed (Delegated to subclass implementation)
        v_max = self._get_max_wave_speed(state)
        
        # 2. Apply DG Formula
        # N refers to polynomial order. Checks if self.N is available.
        N = getattr(self, 'N', 1) # Default to 1 if not set (avoid division by zero)
        if N < 1: N = 1
        
        # Formula: dt = (CFL * 2.0 * R) / (2.0 * v_max * (N ** 2))
        # Simplifies to: CFL * R / (v_max * N**2). 
        # Note: The '2.0' factors cancel out, but we keep the structure clear.
        # Ensure v_max is not zero to avoid error
        if v_max < 1e-12:
            v_max = 1.0 # Fallback for static state
            
        R = self.config.get('R', 1.0)
        dt = float(cfl * R / (v_max * (N**2)))
        return dt

    def _get_max_wave_speed(self, state) -> float:
        """
        Calculate the maximum characteristic wave speed across the domain.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _get_max_wave_speed")

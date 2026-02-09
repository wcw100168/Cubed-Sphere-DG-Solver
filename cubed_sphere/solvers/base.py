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

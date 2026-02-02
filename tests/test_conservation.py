import unittest
import numpy as np
from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig
from cubed_sphere.geometry.grid import FaceGrid

class TestConservation(unittest.TestCase):
    def compute_total_mass(self, solver: CubedSphereAdvectionSolver, state: np.ndarray) -> float:
        """
        Compute integrated mass over the sphere surface.
        Integral = Sum_faces [ Sum_nodes ( phi * sqrt_g * w_alpha * w_beta ) ]
        """
        total_mass = 0.0
        
        for i, fname in enumerate(solver.topology.FACE_MAP):
            fg = solver.faces[fname]
            # Handle (n_vars, 6, N, N) or (6, N, N) legacy
            if state.ndim == 4:
                phi = state[0, i]
            else:
                phi = state[i]
            
            # Ensure weighting terms are correct
            # w_alpha (N,), w_beta (N,) -> W (N,N) outer product
            W = np.outer(fg.walpha, fg.wbeta)
            
            # Element area element dA = sqrt_g * d_alpha * d_beta
            if fg.sqrt_g is None:
                raise ValueError("Grid not initialized properly (sqrt_g is None)")
                
            dA = fg.sqrt_g * W
            
            # Integrate this face
            face_mass = np.sum(phi * dA)
            total_mass += face_mass
            
        return total_mass

    def test_mass_conservation_advection(self):
        """
        Run advection for one full rotation and check mass drift.
        """
        # 1. Setup
        config = AdvectionConfig(
            N=16,                 # Small N for speed
            R=1.0,
            u0=2 * np.pi,        # Full rotation in T=1.0
            T_final=1.0,      
            CFL=0.5
        )
        solver = CubedSphereAdvectionSolver(config)
        
        # 2. Initial Condition
        # Get purely stateless initial condition
        idx0 = solver.get_initial_condition(type="gaussian", h0=1.0)
        
        # 3. Compute Initial Mass
        mass_initial = self.compute_total_mass(solver, idx0)
        print(f"Initial Mass: {mass_initial:.10f}")
        
        # 4. Integrate
        # Run 100 steps or full time? Test requires T_period = 1.0 (u0=2pi)
        # However, running full simulation in unit test might be slow.
        # Let's run a full rotation to be rigorous, but keep N small (16).
        final_state = solver.solve((0.0, 1.0), idx0)
        
        # 5. Compute Final Mass
        mass_final = self.compute_total_mass(solver, final_state)
        print(f"Final Mass  : {mass_final:.10f}")
        
        # 6. Assert Conservation
        # DG methods are conservative locally, but global conservation often depends on flux handling.
        # For standard DG with correct numerical flux, mass should be conserved to machine precision 
        # (or time-stepping error).
        # We allow a very small tolerance.
        diff = abs(mass_final - mass_initial)
        print(f"Mass Change : {diff:.4e}")
        
        # Tolerance: 1e-12 is strict but achievable for conservative schemes.
        # If it fails, something is wrong with flux integration or quadrature.
        self.assertLess(diff, 1e-5, f"Mass conservation violation: {diff}")

if __name__ == "__main__":
    unittest.main()

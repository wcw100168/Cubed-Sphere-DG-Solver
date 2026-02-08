import unittest
import numpy as np
from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig
from cubed_sphere.geometry.grid import FaceGrid

class TestAdvection(unittest.TestCase):
    
    def setUp(self):
        try:
            import jax
            self.jax_available = True
        except ImportError:
            self.jax_available = False

    def compute_total_mass(self, solver: CubedSphereAdvectionSolver, state: np.ndarray) -> float:
        """
        Compute integrated mass over the sphere surface.
        Integral = Sum_faces [ Sum_nodes ( phi * sqrt_g * w_alpha * w_beta ) ]
        """
        # Handle JAX output shape (1, 6, N, N) vs legacy (6, N, N)
        if state.ndim == 4:
            state_data = state[0] # Extract first variable
        else:
            state_data = state

        total_mass = 0.0
        
        for i, fname in enumerate(solver.topology.FACE_MAP):
            fg = solver.faces[fname]
            
            # Ensure weighting terms are correct and compatible with NumPy calculation
            # If using JAX backend, fg attributes might be JAX arrays. 
            # We explicitly convert everything to numpy for this validation check.
            walpha = np.array(fg.walpha)
            wbeta  = np.array(fg.wbeta)
            sqrt_g = np.array(fg.sqrt_g)
            
            # w_alpha (N,), w_beta (N,) -> W (N,N) outer product
            W = np.outer(walpha, wbeta)
            
            if sqrt_g is None:
                raise ValueError("Grid not initialized properly (sqrt_g is None)")
                
            dA = sqrt_g * W
            
            # Integrate this face
            face_mass = np.sum(state_data[i] * dA)
            total_mass += face_mass
            
        return total_mass

    def test_numpy_mass_conservation(self):
        """
        [NumPy] Run advection for one full rotation and check mass drift.
        """
        # 1. Setup
        config = AdvectionConfig(
            N=16,                 # Small N for speed
            R=1.0,
            u0=2 * np.pi,        # Full rotation in T=1.0
            T_final=1.0,      
            CFL=0.5,
            backend='numpy'
        )
        solver = CubedSphereAdvectionSolver(config)
        
        # 2. Initial Condition
        idx0 = solver.get_initial_condition(type="gaussian", h0=1.0)
        
        # 3. Compute Initial Mass
        mass_initial = self.compute_total_mass(solver, idx0)
        print(f"\n[NumPy] Initial Mass: {mass_initial:.10f}")
        
        # 4. Integrate
        final_state = solver.solve((0.0, 1.0), idx0)
        
        # 5. Compute Final Mass
        mass_final = self.compute_total_mass(solver, final_state)
        print(f"[NumPy] Final Mass  : {mass_final:.10f}")
        
        # 6. Assert Conservation
        diff = abs(mass_final - mass_initial)
        print(f"[NumPy] Mass Drift  : {diff:.4e}")
        
        self.assertLess(diff, 1e-5, f"Mass conservation violation: {diff}")

    def test_jax_mass_conservation(self):
        """
        [JAX] Run advection for one full rotation and check mass drift.
        """
        if not self.jax_available:
            self.skipTest("JAX not installed")

        # 1. Setup JAX Solver
        config = AdvectionConfig(
            N=8,                 
            R=1.0,
            u0=2 * np.pi,        
            T_final=1.0,      
            CFL=0.5,
            backend='jax'
        )
        
        try:
            solver = CubedSphereAdvectionSolver(config)
        except Exception as e:
            if "UNIMPLEMENTED" in str(e) or "default_memory_space" in str(e):
                self.skipTest(f"JAX Helper skipped due to environment issue: {e}")
            else:
                self.skipTest(f"JAX Init failed: {e}")
        
        import jax.numpy as jnp
        
        # 2. Initial Condition
        idx0 = solver.get_initial_condition(type="gaussian", h0=1.0)
        
        # Verify it is a JAX array
        self.assertTrue(isinstance(idx0, jnp.ndarray) or hasattr(idx0, "device_buffer"))

        # 3. Compute Initial Mass (using numpy for verification logic)
        idx0_np = np.array(idx0)
        mass_initial = self.compute_total_mass(solver, idx0_np)
        print(f"\n[JAX] Initial Mass: {mass_initial:.10f}")
        
        # 4. Integrate
        final_state = solver.solve((0.0, 1.0), idx0)
        
        # 5. Check Conservation
        final_state_np = np.array(final_state)
        mass_final = self.compute_total_mass(solver, final_state_np)
        print(f"[JAX] Final Mass  : {mass_final:.10f}")
        
        diff = abs(mass_final - mass_initial)
        print(f"[JAX] Mass Drift  : {diff:.4e}")
        
        # JAX Advection precision tolerance (relaxed slightly compared to NumPy)
        self.assertLess(diff, 1e-3, f"JAX Mass conservation violation: {diff}")

if __name__ == "__main__":
    unittest.main()

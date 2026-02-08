import unittest
import numpy as np
import os
from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig

class TestConsistency(unittest.TestCase):
    
    def setUp(self):
        try:
            import jax
            # Force x64 for consistency check
            jax.config.update("jax_enable_x64", True)
            self.jax_available = True
            
            # Simple Config (Removed dt because AdvectionConfig might not take it directly or its ignored)
            self.config_dict = {
                'N': 8,
                'R': 1.0,
                'u0': 2.0 * np.pi,
                'T_final': 0.1,
                'CFL': 0.1
            }
            
        except ImportError:
            self.jax_available = False

    def test_advection_step_consistency(self):
        """
        Verify that NumPy and JAX backends produce identical results (to tight tolerance)
        for a single RK time step.
        """
        if not self.jax_available:
            self.skipTest("JAX not installed")
            
        # 1. Initialize NumPy Solver
        config_np = AdvectionConfig(**self.config_dict, backend='numpy')
        solver_np = CubedSphereAdvectionSolver(config_np)
        
        # 2. Initialize JAX Solver
        config_jax = AdvectionConfig(**self.config_dict, backend='jax')
        try:
            solver_jax = CubedSphereAdvectionSolver(config_jax)
        except Exception as e:
            self.skipTest(f"JAX Init failed: {e}")
            
        # 3. Common Initial Condition
        # Use simple Gaussian
        initial_condition_np = solver_np.get_initial_condition("gaussian", h0=1.0)
        
        # JAX solver needs JAX array usually, but it might auto-convert.
        # Let's verify initial state.
        import jax.numpy as jnp
        initial_condition_jax = jnp.array(initial_condition_np)
        
        # 4. Run One Step using solve()
        # The AdvectionSolver doesn't expose a public 'step' method, but 'solve' runs the loop.
        # We set T_final to be small so that only one step (or very few identical steps) is taken.
        dt = 0.001
        
        # NumPy Step
        # solver.solve returns the final state
        state_np_next = solver_np.solve((0.0, dt), initial_condition_np)
        
        # JAX Step
        state_jax_next_d = solver_jax.solve((0.0, dt), initial_condition_jax)
        state_jax_next = np.array(state_jax_next_d)
        state_jax_next = np.array(state_jax_next_d)
        
        # 5. Assert Consistency
        # Handle Potential Shape Mismatch (JAX: 1,6,N,N vs NumPy: 6,N,N)
        if state_jax_next.ndim == 4 and state_np_next.ndim == 3:
            state_jax_next = state_jax_next[0]
            
        diff = np.max(np.abs(state_np_next - state_jax_next))
        print(f"\nMax Consistency Diff (NumPy vs JAX): {diff:.6e}")
        
        # Tolerance: 
        # With x64, we expect near machine epsilon agreement if algorithms are identical.
        # Python vs JAX XLA operations order might differ slightly.
        # 1e-14 is usually safe for x64 bit-identical logic.
        np.testing.assert_allclose(state_np_next, state_jax_next, atol=1e-14, rtol=1e-14)

if __name__ == "__main__":
    unittest.main()

import unittest
import os
# Force JAX to use CPU interface to avoid Metal/MPS experimental errors during testing
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
# Enable 64-bit precision for mass conservation checks
jax.config.update("jax_enable_x64", True)

import numpy as np
import dataclasses
from cubed_sphere.solvers.swe import CubedSphereSWE, SWEConfig

class TestSWEIntegrationJax(unittest.TestCase):
    
    def setUp(self):
        # Common configuration
        self.config = SWEConfig(
            N=8,
            CFL=0.1,
            backend='jax',
            filter_order=8,
            n_vars=3,
            H_avg=10000.0,
            dt=10.0,
            R=6.37122e6,
            Omega=7.292e-5,
            gravity=9.80616
        )

    def test_backend_selection(self):
        """
        Verify that the facade correctly selects the JAX backend.
        """
        solver = CubedSphereSWE(self.config)
        from cubed_sphere.solvers.swe_jax import CubedSphereSWEJax
        self.assertIsInstance(solver._impl, CubedSphereSWEJax, 
                             "Implementation should be CubedSphereSWEJax")

    def test_case2_steady_state_jax(self):
        """
        Verify Williamson Case 2 with JAX backend.
        """
        solver = CubedSphereSWE(self.config)
        
        # Physics Parameters for Case 2
        u0 = 2.0 * np.pi * self.config.R / (12.0 * 24.0 * 3600.0) 
        
        # Initialize State: (3, 6, N+1, N+1) (Variables, Faces, Xi, Eta)
        grid_size = self.config.N + 1
        state = np.zeros((3, 6, grid_size, grid_size))
        
        R = self.config.R
        Omega = self.config.Omega
        g = self.config.gravity
        
        # Helper to convert to contravariant (Simple version)
        def get_contravariant_simple(fg, u_sph, v_sph):
            # For Case 2 (Zonal Flow), u_sph matches Latitude lines.
            # We can use the solver's internal geometry or just approximate.
            # But tests should match `swe_integration.py` logic.
            # Let's use the implementation's metrics which we can access.
            
            # Note: fg in JAX solver has JAX arrays. We need to act carefully.
            # Convert jax arrays to numpy for setup
            
            lam = np.array(fg.lon)
            theta = np.array(fg.lat)
            g1_vec = np.array(fg.g1_vec)
            g2_vec = np.array(fg.g2_vec)
            g_inv = np.array(fg.g_inv)
            
            e_lam_x = -np.sin(lam)
            e_lam_y = np.cos(lam)
            e_lam_z = np.zeros_like(lam)
            
            e_th_x = -np.sin(theta)*np.cos(lam)
            e_th_y = -np.sin(theta)*np.sin(lam)
            e_th_z = np.cos(theta)
            
            Vx = u_sph * e_lam_x + v_sph * e_th_x
            Vy = u_sph * e_lam_y + v_sph * e_th_y
            Vz = u_sph * e_lam_z + v_sph * e_th_z
            
            V_dot_g1 = Vx * g1_vec[..., 0] + Vy * g1_vec[..., 1] + Vz * g1_vec[..., 2]
            V_dot_g2 = Vx * g2_vec[..., 0] + Vy * g2_vec[..., 1] + Vz * g2_vec[..., 2]
            
            u1 = g_inv[..., 0, 0] * V_dot_g1 + g_inv[..., 0, 1] * V_dot_g2
            u2 = g_inv[..., 1, 0] * V_dot_g1 + g_inv[..., 1, 1] * V_dot_g2
            
            return u1, u2

        for i, face_name in enumerate(solver._impl.topology.FACE_MAP):
            fg = solver._impl.faces[face_name]
            theta = np.array(fg.lat)
            
            u_sph = u0 * np.cos(theta)
            v_sph = np.zeros_like(theta)
            h = self.config.H_avg - (R * Omega * u0 + 0.5 * u0**2) * (np.sin(theta)**2) / g
            
            u1, u2 = get_contravariant_simple(fg, u_sph, v_sph)
            
            # Correct Layout: (Var, Face) 
            state[0, i] = h
            state[1, i] = u1
            state[2, i] = u2
            
        def calc_mass(s):
            mass = 0.0
            # s is likely numpy array here since we are in test setup/verification
            for k, face_name in enumerate(solver._impl.topology.FACE_MAP):
                fg = solver._impl.faces[face_name]
                # In JAX implementation, fg fields are JAX arrays. Convert to numpy for test math.
                sqrt_g = np.array(fg.sqrt_g)
                walpha = np.array(fg.walpha)
                wbeta = np.array(fg.wbeta)
                w = walpha[:, None] * wbeta[None, :]
                # Layout: s[Var, Face, ...]
                mass += np.sum(s[0, k] * sqrt_g * w)
            return mass

        initial_mass = calc_mass(state)

        # Run Simulation
        dt = 10.0
        t = 0.0
        # Reduce n_steps to ensure stability in testing environment or check logic
        # If dt=10 and stable, 20 steps is fine.
        n_steps = 20 
        
        current_state = solver.solve((0.0, dt * n_steps), state) 
        # Note: solve returns numpy array per our implementation
        
        # Analysis
        final_mass = calc_mass(current_state)
        mass_diff = abs(final_mass - initial_mass)
        # Check Error dims: state is (3, 6, ...)
        h_error = np.max(np.abs(current_state[0] - state[0]))
        
        print(f"\nJAX Integration Results (N={self.config.N}, Steps={n_steps}):")
        print(f"  Mass Error (Abs): {mass_diff:.5e}")
        print(f"  Mass Error (Rel): {mass_diff/initial_mass:.5e}")
        print(f"  H L_inf Error: {h_error:.5e}")
        
        self.assertLess(mass_diff / initial_mass, 1e-14) # Relative Mass Conservation
        self.assertLess(h_error, 1e-4)

if __name__ == '__main__':
    unittest.main()

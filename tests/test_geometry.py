import unittest
import numpy as np
import math
from cubed_sphere.geometry.grid import CubedSphereEquiangular, CubedSphereTopology

class TestGeometry(unittest.TestCase):
    
    def test_surface_area(self):
        """
        Verify that the integral of sqrt_g over the sphere equals 4*pi*R^2.
        """
        R = 6.37122e6
        N_poly = 32
        num_nodes = N_poly + 1
        
        geo = CubedSphereEquiangular(R=R)
        topo = CubedSphereTopology()
        
        total_area = 0.0
        
        for fname in topo.FACE_MAP:
            fg = geo.generate_face(num_nodes, fname)
            
            # Integration weights
            w_alpha = fg.walpha # (num_nodes,)
            w_beta = fg.wbeta   # (num_nodes,)
            W = np.outer(w_alpha, w_beta) # (num_nodes, num_nodes)
            
            # Area Element
            # sqrt_g for equiangular coordinates
            diff_area = fg.sqrt_g * W
            
            total_area += np.sum(diff_area)
            
        expected_area = 4.0 * math.pi * R**2
        
        # Check Error
        error = abs(total_area - expected_area)
        relative_error = error / expected_area
        
        print(f"\nSurface Area Integration:")
        print(f"Computed: {total_area:.6e}")
        print(f"Expected: {expected_area:.6e}")
        print(f"Rel Error: {relative_error:.6e}")
        
        # Spectral accuracy should give very low error for N=32
        self.assertLess(relative_error, 1e-12)

    def test_topology_connectivity(self):
        """
        Verify the connectivity table for Face 0 against known Cubed Sphere layout.
        Standard Layout:
          N
        W 0 E
          S
        
        Face 0 Neighbors:
        - West (Side 0) -> Face 3 (Side 1) (Matches Panel 3 East)
        - East (Side 1) -> Face 1 (Side 0) (Matches Panel 1 West)
        - South (Side 2) -> Face 5 (Side 3) (Matches Panel 5 North) - wrapped
        - North (Side 3) -> Face 4 (Side 2) (Matches Panel 4 South)
        """
        topo = CubedSphereTopology()
        
        # Face 0 is P1 (Equatorial 1)
        # Check Standard Neighbors
        
        # 1. West Neighbor (Side 0)
        # Expected: Face 3 (P4)
        nbr_face, nbr_side, swap, rev = topo.CONN_TABLE[(0, 0)]
        self.assertEqual(nbr_face, 3, "Face 0 West should be Face 3")
        
        # 2. East Neighbor (Side 1)
        # Expected: Face 1 (P2)
        nbr_face, nbr_side, swap, rev = topo.CONN_TABLE[(0, 1)]
        self.assertEqual(nbr_face, 1, "Face 0 East should be Face 1")
        self.assertEqual(nbr_side, 0, "Face 0 East should connect to Face 1 West")
        
        # 3. North Neighbor (Side 3)
        # Expected: Face 4 (P5 - Top)
        nbr_face, nbr_side, swap, rev = topo.CONN_TABLE[(0, 3)]
        self.assertEqual(nbr_face, 4, "Face 0 North should be Face 4")
        
        # 4. South Neighbor (Side 2)
        # Expected: Face 5 (P6 - Bottom)
        nbr_face, nbr_side, swap, rev = topo.CONN_TABLE[(0, 2)]
        self.assertEqual(nbr_face, 5, "Face 0 South should be Face 5")

if __name__ == "__main__":
    unittest.main()

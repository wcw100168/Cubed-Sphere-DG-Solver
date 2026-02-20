import sys
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import jax
import jax.numpy as jnp
import numpy as np
from cubed_sphere.solvers.swe_jax import CubedSphereSWEJax

def verify_topology():
    print("Initializing solver to build static topology...")
    config = {'N': 4} # Small N for speed
    solver = CubedSphereSWEJax(config)
    
    indices = solver.neighbor_indices
    print(f"\nNeighbor Index Array Shape: {indices.shape}")
    print("Expected: (6, 4, 3) -> (Face, Side, [NbrFace, NbrSide, Reverse])")
    
    # Check shape
    assert indices.shape == (6, 4, 3)
    
    print("\nSample Connectivity (Face 0):")
    # Face 0 often connects to [West:4, East:1, South:5, North:2] or similar depending on layout
    names = ["West", "East", "South", "North"]
    for side in range(4):
        nbr_face = indices[0, side, 0]
        nbr_side = indices[0, side, 1]
        reverse = indices[0, side, 2]
        print(f"  Side {side} ({names[side]}): Connects to Face {nbr_face}, Side {nbr_side}, Reverse={reverse}")
        
    print("\nFull Array Dump (First 2 faces):")
    print(indices[:2])

if __name__ == "__main__":
    verify_topology()

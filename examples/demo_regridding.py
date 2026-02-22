import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig
from cubed_sphere.utils.ic_builder import build_from_latlon_grid


def synthetic_coarse_field(lat_deg, lon_deg):
    # Localized bump centered near (20N, 40E)
    lat0, lon0 = 20.0, 40.0
    width = 15.0
    dlat = lat_deg - lat0
    dlon = (lon_deg - lon0)
    val = np.exp(-((dlat**2 + dlon**2) / width**2))
    return val


def main():
    # 5-degree coarse grid
    lat_1d = np.arange(-90, 90 + 5, 5)
    lon_1d = np.arange(0, 360, 5)
    lon_mesh, lat_mesh = np.meshgrid(lon_1d, lat_1d)
    coarse_data = synthetic_coarse_field(lat_mesh, lon_mesh)

    solver = CubedSphereAdvectionSolver(AdvectionConfig(N=32, backend="numpy", T_final=0.0))

    state = build_from_latlon_grid(solver, lat_1d, lon_1d, coarse_data)
    solver.validate_state(state)

    # Plot Face 0
    face0 = state[0, 0]
    plt.figure(figsize=(6, 5))
    plt.contourf(face0, levels=30, cmap="viridis")
    plt.colorbar(label="Interpolated value")
    plt.title("Face 0 regridded scalar field")
    plt.tight_layout()
    plt.savefig("demo_regridding_face0.png")
    print("Saved demo_regridding_face0.png")


if __name__ == "__main__":
    main()

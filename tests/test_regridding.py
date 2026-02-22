import numpy as np
import pytest

from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig
from cubed_sphere.utils.ic_builder import build_from_latlon_grid


def test_regrid_constant_field():
    solver = CubedSphereAdvectionSolver(AdvectionConfig(N=3, backend="numpy", n_vars=1))
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(0, 360, 9, endpoint=False)
    data = np.full((len(lat), len(lon)), 5.0)

    state = build_from_latlon_grid(solver, lat, lon, data)

    expected_shape = (solver.cfg.n_vars, 6, solver.cfg.N + 1, solver.cfg.N + 1)
    assert state.shape == expected_shape
    assert np.allclose(state, 5.0)
    solver.validate_state(state)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

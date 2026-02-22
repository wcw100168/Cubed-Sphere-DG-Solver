import numpy as np
import pytest

from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig
from cubed_sphere.utils.ic_builder import build_from_function


def test_build_from_function_constant_one():
    solver = CubedSphereAdvectionSolver(AdvectionConfig(N=3, backend="numpy", n_vars=2))
    state = build_from_function(solver, lambda lon, lat: np.ones_like(lon))
    expected_shape = (solver.cfg.n_vars, 6, solver.cfg.N + 1, solver.cfg.N + 1)
    assert state.shape == expected_shape
    assert np.all(state == 1.0)
    # Should also pass solver validation
    solver.validate_state(state)


def test_build_from_function_var_idx_and_sqrt_g():
    solver = CubedSphereAdvectionSolver(AdvectionConfig(N=2, backend="numpy", n_vars=2))
    state = build_from_function(solver, lambda lon, lat: lon * 0 + 2.0, var_idx=1)
    # Only var_idx=1 should be filled; var 0 should remain zeros
    assert np.all(state[0] == 0.0)
    assert np.all(state[1] == 2.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

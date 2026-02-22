import numpy as np
import pytest

from cubed_sphere.solvers import CubedSphereAdvectionSolver, AdvectionConfig
from cubed_sphere.solvers.swe import CubedSphereSWE, SWEConfig


def _make_advection_solver():
    return CubedSphereAdvectionSolver(AdvectionConfig(N=4, backend="numpy", n_vars=1))


def _make_swe_solver():
    return CubedSphereSWE(SWEConfig(N=2, backend="numpy"))


def test_wrong_poly_order_shape():
    solver = _make_advection_solver()
    # Intentionally drop one node (use N instead of N+1)
    bad_shape_state = np.zeros((solver.cfg.n_vars, 6, solver.cfg.N, solver.cfg.N))
    with pytest.raises(ValueError, match="Expected state shape"):
        solver.validate_state(bad_shape_state)


def test_missing_variables():
    solver = _make_swe_solver()
    num_nodes = solver.swe_config.N + 1
    # Provide only one variable instead of three
    bad_var_state = np.zeros((1, 6, num_nodes, num_nodes))
    with pytest.raises(ValueError, match="Expected state shape"):
        solver.validate_state(bad_var_state)


def test_nan_injection():
    solver = _make_advection_solver()
    num_nodes = solver.cfg.N + 1
    state_with_nan = np.zeros((solver.cfg.n_vars, 6, num_nodes, num_nodes))
    state_with_nan[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        solver.validate_state(state_with_nan)


def test_valid_state_passes():
    solver = _make_advection_solver()
    num_nodes = solver.cfg.N + 1
    good_state = np.ones((solver.cfg.n_vars, 6, num_nodes, num_nodes))
    # Should not raise
    solver.validate_state(good_state)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

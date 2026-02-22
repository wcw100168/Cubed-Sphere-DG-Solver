"""Utilities for building initial conditions from user functions.

This module lets users supply simple callables f(lon, lat) without dealing with
per-face loops or geometry internals. It returns a NumPy array in the solver's
expected layout (n_vars, 6, N+1, N+1), ready for validate_state().
"""

from typing import Callable, Optional
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _resolve_config(solver):
    """Return a config dict-like object from solver (supports different solver APIs)."""
    if hasattr(solver, "cfg"):
        return solver.cfg
    if hasattr(solver, "swe_config"):
        return solver.swe_config
    return solver.config


def _get_faces_and_geometry(solver):
    """Return (faces, topology, geometry) regardless of solver backend facade."""
    faces = getattr(solver, "faces", None)
    topology = getattr(solver, "topology", None)
    geometry = getattr(solver, "geometry", None)

    # SWE facade keeps faces/topology inside _impl
    if faces is None and hasattr(solver, "_impl"):
        faces = getattr(solver._impl, "faces", None)
    if topology is None and hasattr(solver, "_impl"):
        topology = getattr(solver._impl, "topology", None)
    if geometry is None and hasattr(solver, "geometry", None):
        geometry = solver.geometry

    if faces is None or topology is None or geometry is None:
        raise ValueError("Solver does not expose faces/topology/geometry needed for IC building.")

    return faces, topology, geometry


def build_from_function(
    solver,
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    var_idx: Optional[int] = None,
    multiply_by_sqrt_g: bool = False,
    dtype=None,
):
    """Construct an initial state tensor by evaluating a callable on lon/lat.

    Args:
        solver: Solver instance (advection or SWE) exposing faces/topology/geometry.
        func: Callable f(lon, lat) returning scalar field on radians arrays.
        var_idx: Optional variable index to write into. If None, fills all vars.
        multiply_by_sqrt_g: If True, multiply results by Jacobian sqrt_g per face
            (useful for SWE mass initialization).
        dtype: Optional dtype for allocation; falls back to float64.

    Returns:
        np.ndarray shaped (n_vars, 6, N+1, N+1) with evaluated values.
    """

    cfg = _resolve_config(solver)
    n_vars = cfg.n_vars if hasattr(cfg, "n_vars") else cfg.get("n_vars", 1)
    N = cfg.N if hasattr(cfg, "N") else cfg.get("N")
    if N is None:
        raise ValueError("Solver config missing N; cannot build initial condition.")

    faces, topology, geometry = _get_faces_and_geometry(solver)

    num_nodes = N + 1
    arr = np.zeros((n_vars, 6, num_nodes, num_nodes), dtype=dtype if dtype is not None else float)

    for face_idx, fname in enumerate(topology.FACE_MAP):
        fg = faces[fname]
        # Get lon/lat; some faces store lon/lat, otherwise derive from XYZ.
        if hasattr(fg, "lon") and hasattr(fg, "lat"):
            lon, lat = np.array(fg.lon), np.array(fg.lat)
        else:
            X, Y, Z = np.array(fg.X), np.array(fg.Y), np.array(fg.Z)
            lon, lat = geometry.lonlat_from_xyz(X, Y, Z)

        vals = func(lon, lat)
        if multiply_by_sqrt_g:
            vals = vals * fg.sqrt_g

        if var_idx is None:
            arr[:, face_idx, :, :] = vals
        else:
            if var_idx < 0 or var_idx >= n_vars:
                raise ValueError(f"var_idx {var_idx} out of range for n_vars={n_vars}.")
            arr[var_idx, face_idx, :, :] = vals

    return arr


def _ensure_sorted(name: str, arr: np.ndarray) -> None:
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got shape {arr.shape}.")
    if not np.all(np.diff(arr) > 0):
        raise ValueError(f"{name} must be strictly increasing for interpolation; got non-monotonic values.")


def _wrap_longitudes(target_lon_deg: np.ndarray, src_lon: np.ndarray) -> np.ndarray:
    src_min, src_max = float(np.min(src_lon)), float(np.max(src_lon))
    target = np.array(target_lon_deg)
    if src_min >= 0 and src_max > 180:  # source likely 0..360
        target = np.where(target < 0, target + 360.0, target)
    else:  # source likely -180..180
        target = np.where(target > 180, target - 360.0, target)
    return target


def build_from_latlon_grid(
    solver,
    lat: np.ndarray,
    lon: np.ndarray,
    data: np.ndarray,
    var_idx: Optional[int] = None,
    multiply_by_sqrt_g: bool = False,
    method: str = "linear",
    dtype=None,
):
    """Regrid a scalar lat-lon field onto cubed-sphere LGL nodes.

    Args:
        solver: Solver instance (advection or SWE).
        lat: 1D latitudes in degrees.
        lon: 1D longitudes in degrees.
        data: 2D array shaped (len(lat), len(lon)).
        var_idx: Optional variable index to fill; None fills all.
        multiply_by_sqrt_g: Multiply by Jacobian per face (for mass variables).
        method: Interpolation method for RegularGridInterpolator.
        dtype: Optional dtype for allocation.
    """

    cfg = _resolve_config(solver)
    n_vars = cfg.n_vars if hasattr(cfg, "n_vars") else cfg.get("n_vars", 1)
    N = cfg.N if hasattr(cfg, "N") else cfg.get("N")
    if N is None:
        raise ValueError("Solver config missing N; cannot build initial condition.")

    _ensure_sorted("lat", np.asarray(lat))
    _ensure_sorted("lon", np.asarray(lon))
    if data.shape != (len(lat), len(lon)):
        raise ValueError(f"data shape {data.shape} does not match (len(lat), len(lon))={(len(lat), len(lon))}.")

    interp = RegularGridInterpolator(
        (np.asarray(lat), np.asarray(lon)),
        np.asarray(data),
        bounds_error=False,
        fill_value=None,
        method=method,
    )

    faces, topology, geometry = _get_faces_and_geometry(solver)
    num_nodes = N + 1
    arr = np.zeros((n_vars, 6, num_nodes, num_nodes), dtype=dtype if dtype is not None else float)

    for face_idx, fname in enumerate(topology.FACE_MAP):
        fg = faces[fname]
        if hasattr(fg, "lon") and hasattr(fg, "lat"):
            lon_rad, lat_rad = np.array(fg.lon), np.array(fg.lat)
        else:
            X, Y, Z = np.array(fg.X), np.array(fg.Y), np.array(fg.Z)
            lon_rad, lat_rad = geometry.lonlat_from_xyz(X, Y, Z)

        lon_deg = np.degrees(lon_rad)
        lat_deg = np.degrees(lat_rad)
        lon_deg = _wrap_longitudes(lon_deg, np.asarray(lon))

        query = np.stack([lat_deg.flatten(), lon_deg.flatten()], axis=-1)
        vals = interp(query).reshape(num_nodes, num_nodes)

        if multiply_by_sqrt_g:
            vals = vals * fg.sqrt_g

        if var_idx is None:
            arr[:, face_idx, :, :] = vals
        else:
            if var_idx < 0 or var_idx >= n_vars:
                raise ValueError(f"var_idx {var_idx} out of range for n_vars={n_vars}.")
            arr[var_idx, face_idx, :, :] = vals

    return arr

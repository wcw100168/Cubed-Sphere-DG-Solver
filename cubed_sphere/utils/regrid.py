import numpy as np
from scipy.interpolate import RegularGridInterpolator
from cubed_sphere.geometry.grid import FaceGrid, CubedSphereEquiangular

def from_latlon_grid(
    target_grid: FaceGrid, 
    src_lat: np.ndarray, 
    src_lon: np.ndarray, 
    src_data: np.ndarray
) -> np.ndarray:
    """
    Interpolate data from a regular Lat/Lon grid to a Cubed Sphere FaceGrid.
    
    Parameters
    ----------
    target_grid : FaceGrid
        The target face grid (contains Cartesian coordinates X, Y, Z).
    src_lat : np.ndarray
        1D array of source latitudes in Degrees.
    src_lon : np.ndarray
        1D array of source longitudes in Degrees.
    src_data : np.ndarray
        2D array of data on the source grid (shape: (n_lat, n_lon) or similar).
        
    Returns
    -------
    np.ndarray
        Interpolated data on the target grid (shape N x N).
    """

    # 1. Input Validation & Preparation
    # Ensure source grid is sorted if needed, but RegularGridInterpolator handles unsorted? 
    # Actually RegularGridInterpolator assumes sorted grid points. 
    # Let's assume user provides typical sorted data, or let scipy handle errors.
    
    # 2. Create Interpolator
    # Note: RegularGridInterpolator expects (points_dim1, points_dim2, ...)
    # If src_data is (lat, lon), then points should be (src_lat, src_lon)
    
    # Handle periodicity in Longitude?
    # Some datasets are 0..360, some -180..180.
    # The user instruction says "Handle the periodicity... or assume...".
    # For now, we trust the interpolator unless boundaries are an issue.
    # bounds_error=False, fill_value=None will just extrapolate or fill.
    
    interp = RegularGridInterpolator(
        (src_lat, src_lon), 
        src_data, 
        bounds_error=False, 
        fill_value=None,
        method='linear' 
    )
    
    # 3. Get Target Coordinates (Radians)
    # We must ensure we have numpy arrays for coordinates
    X = np.array(target_grid.X)
    Y = np.array(target_grid.Y)
    Z = np.array(target_grid.Z)
    
    # Convert to Lon/Lat (Radians)
    # Re-use the static method from geometry class
    lam, theta = CubedSphereEquiangular.lonlat_from_xyz(X, Y, Z)
    
    # 4. Convert Target to Degrees for Query
    target_lat_deg = np.degrees(theta)
    target_lon_deg = np.degrees(lam)
    
    # 5. Handle Longitude convention mismatch (0..360 vs -180..180)
    # Standardize target to match source convention if possible.
    # Check source range
    src_lon_min, src_lon_max = np.min(src_lon), np.max(src_lon)
    
    if src_lon_min >= 0 and src_lon_max > 180:
        # Source is likely 0..360
        target_lon_deg = np.where(target_lon_deg < 0, target_lon_deg + 360, target_lon_deg)
    elif src_lon_min < 0:
        # Source is likely -180..180
        # target_lon_deg is naturally -180..180 from arctan2
        pass
        
    # 6. Query Interpolator
    # We need to flatten the target grid for query points -> (N_points, 2)
    query_points = np.stack([target_lat_deg.flatten(), target_lon_deg.flatten()], axis=-1)
    
    result_flat = interp(query_points)
    
    # 7. Reshape back to FaceGrid shape
    return result_flat.reshape(target_grid.X.shape)

def latlon_to_cartesian_velocity(u_lon, v_lat, lam, theta):
    """
    Convert spherical velocity components to 3D Cartesian velocity.
    
    v_x = -u sin(lambda) - v sin(theta) cos(lambda)
    v_y =  u cos(lambda) - v sin(theta) sin(lambda)
    v_z =  v cos(theta)
    
    Parameters
    ----------
    u_lon : np.ndarray
        Zonal velocity (East-West).
    v_lat : np.ndarray
        Meridional velocity (North-South).
    lam : np.ndarray
        Longitude (radians).
    theta : np.ndarray
        Latitude (radians).
        
    Returns
    -------
    vx, vy, vz : np.ndarray
    """
    sin_lam, cos_lam = np.sin(lam), np.cos(lam)
    sin_th, cos_th   = np.sin(theta), np.cos(theta)
    
    vx = -u_lon * sin_lam - v_lat * sin_th * cos_lam
    vy =  u_lon * cos_lam - v_lat * sin_th * sin_lam
    vz =  v_lat * cos_th
    
    return vx, vy, vz

def cartesian_velocity_to_spherical(vx, vy, vz, lam, theta):
    """
    Project 3D Cartesian velocity back to spherical components (East, North).
    Used after interpolation on the target grid.
    
    u_lon = V . e_lambda
    v_lat = V . e_theta
    
    e_lambda = [-sin(lam), cos(lam), 0]
    e_theta  = [-sin(theta)cos(lam), -sin(theta)sin(lam), cos(theta)]
    """
    sin_lam, cos_lam = np.sin(lam), np.cos(lam)
    sin_th, cos_th   = np.sin(theta), np.cos(theta)
    
    # Dot products
    u_lon = -vx * sin_lam + vy * cos_lam
    v_lat = -vx * sin_th * cos_lam - vy * sin_th * sin_lam + vz * cos_th
    
    return u_lon, v_lat

def from_latlon_vector(
    target_grid: FaceGrid, 
    src_lat: np.ndarray, 
    src_lon: np.ndarray, 
    src_u: np.ndarray,
    src_v: np.ndarray,
    geometry_handler: CubedSphereEquiangular = None
) -> np.ndarray:
    """
    Regrid a vector field from Lat/Lon to Cubed Sphere Contravariant components.
    
    Algorithm:
    1. Convert source (u, v) -> (vx, vy, vz) on source grid.
    2. Interpolate (vx, vy, vz) to target grid points.
    3. Convert target (vx, vy, vz) -> (u_sph, v_sph) on target grid.
    4. Convert (u_sph, v_sph) -> (u1, u2) using geometry handler.
    
    Returns
    -------
    u1, u2 : Tuple[np.ndarray, np.ndarray]
        Contravariant velocity components on the target grid.
    """
    if geometry_handler is None:
        # Fallback if not provided, though inefficient to recreate if R is unknown
        # Assuming unit sphere or relying on caller to provide
        geometry_handler = CubedSphereEquiangular(R=1.0) 

    # 1. Convert Source to Cartesian
    # Meshgrid needed for source coordinate arrays matching data shape
    # src_lat is usually 1D, src_lon 1D. src_u is 2D (lat, lon) usually.
    # We need 2D mesh of coords.
    src_lon_2d, src_lat_2d = np.meshgrid(src_lon, src_lat) # Verify indexing usually xy
    # However, if src_u is (n_lat, n_lon), meshgrid should produce compatible shapes.
    # src_lon_2d (n_lat, n_lon) if indexing='xy' and inputs are (lon, lat)?
    # Let's standardize on inputs: src_lat (N,), src_lon (M,), src_u (N, M)
    # np.meshgrid(src_lon, src_lat) give (N, M) if indexing='xy'? No, (N, M) requires 'ij' if arg1 is lat.
    # Actually: meshgrid(x, y) -> X (M, N), Y (M, N) usually.
    # Let's use broadcasting implicitly or check shapes.
    
    # Better: Perform conversion on 1D arrays if possible? No, u/v depend on both.
    src_lon_rad = np.radians(src_lon_2d)
    src_lat_rad = np.radians(src_lat_2d)
    
    src_vx, src_vy, src_vz = latlon_to_cartesian_velocity(src_u, src_v, src_lon_rad, src_lat_rad)
    
    # 2. Interpolate Cartesian Components
    # Reuse from_latlon_grid logic for scalar interpolation
    # BUT from_latlon_grid takes 1D lat/lon and 2D data.
    # src_vx is 2D. Perfect.
    
    vx_interp = from_latlon_grid(target_grid, src_lat, src_lon, src_vx)
    vy_interp = from_latlon_grid(target_grid, src_lat, src_lon, src_vy)
    vz_interp = from_latlon_grid(target_grid, src_lat, src_lon, src_vz)
    
    # 3. Project to Spherical on Target
    # Get target coordinates in radians
    lam_t, theta_t = CubedSphereEquiangular.lonlat_from_xyz(target_grid.X, target_grid.Y, target_grid.Z)
    
    u_sph_t, v_sph_t = cartesian_velocity_to_spherical(vx_interp, vy_interp, vz_interp, lam_t, theta_t)
    
    # 4. Convert to Contravariant
    u1, u2 = geometry_handler.compute_contravariant_vel(target_grid, u_sph_t, v_sph_t)
    
    return u1, u2

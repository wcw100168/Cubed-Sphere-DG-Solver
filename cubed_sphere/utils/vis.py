import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List, Any

# Define custom colormap
COLORS_LIST = [
    (0.0, "green"),
    (0.5, "yellow"),
    (1.0, "red"),
]
GREEN_RED_CMAP = LinearSegmentedColormap.from_list("green_red", COLORS_LIST)

def animate_from_netcdf(nc_filename: str, output_mp4: str, var_idx: int = 0, fps: int = 10):
    """
    Generate MP4 animation from NetCDF file by Re-generating the Grid geometry.
    Uses netCDF4 directly to avoid xarray issues in some environments.
    """
    import netCDF4 as nc
    from cubed_sphere.geometry.grid import CubedSphereEquiangular, CubedSphereTopology
    
    # Read Logic
    try:
        ds = nc.Dataset(nc_filename, 'r')
        times = ds.variables['time'][:]
        # Check if 'xi' or 'nx' exists
        if 'xi' in ds.dimensions:
            N = ds.dimensions['xi'].size
        else:
            # Fallback or error
            N = ds.variables['state'].shape[3]
            
        # Load data (could be large, but for demo ok)
        data_all = ds.variables['state'][:] # (time, var, face, xi, eta)
        ds.close()
    except Exception as e:
        print(f"Failed to open NetCDF with netCDF4: {e}")
        return

    # Reconstruct Geometry (assuming R=1.0 default)
    # Use the same face definition as Topology
    face_names = CubedSphereTopology.FACE_MAP
    geo = CubedSphereEquiangular(R=1.0)
    faces = {}
    for fname in face_names:
        faces[fname] = geo.generate_face(N, fname)
        
    # Setup Figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Animation: {nc_filename}")
    
    # Pre-compute min/max for constant colorbar
    print("Computing global min/max for colorbar...")
    # data_all is (time, var, face, xi, eta)
    if data_all.ndim == 5:
        sub_data = data_all[:, var_idx, :, :, :]
    else: 
        sub_data = data_all
        
    vmin = np.min(sub_data)
    vmax = np.max(sub_data)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    def init():
        ax.clear()
        ax.set_box_aspect([1, 1, 1])
        return []

    def update(frame):
        ax.clear()
        
        t = times[frame]
        ax.set_title(f"Time: {t:.3f}")
        ax.set_box_aspect([1, 1, 1])
        
        # State at frame
        state_t = sub_data[frame] # (6, N, N)
        
        for i, fname in enumerate(face_names):
            fg = faces[fname]
            data = state_t[i]
            rgba = GREEN_RED_CMAP(norm(data))
            
            ax.plot_surface(fg.X, fg.Y, fg.Z, rstride=1, cstride=1,
                           facecolors=rgba,
                           linewidth=0, antialiased=False, shade=False, alpha=0.8)
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=len(times), init_func=init, blit=False)
    
    print(f"Saving animation to {output_mp4}...")
    ani.save(output_mp4, fps=fps, writer='ffmpeg')
    print("Done.")

def plot_cubed_sphere_state(
    solver: Any, 
    state: np.ndarray, 
    var_idx: int = 0,
    title: str = "Advection on Cubed-Sphere",
    cmap: Any = GREEN_RED_CMAP,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 0.6
):
    """
    Plot the scalar field on the cubed sphere in 3D.
    
    Parameters
    ----------
    solver : CubedSphereAdvectionSolver
        The solver instance containing grid information.
    state : np.ndarray
        The state array. Can be (6, N, N) or (n_vars, 6, N, N).
    var_idx : int
        Index of variable to plot if state is 4D.
    title : str
        Plot title.
    cmap : Colormap
        Matplotlib colormap.
    vmin, vmax : float (Optional)
        Colorbar range. If None, min/max of state will be used.
    alpha : float
        Transparency of the surface.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Handle scalar vs system state
    if state.ndim == 4:
        plot_data = state[var_idx]
    else:
        plot_data = state
    
    if vmin is None: vmin = np.min(plot_data)
    if vmax is None: vmax = np.max(plot_data)
    
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot each face
    for i, fname in enumerate(solver.topology.FACE_MAP):
        fg = solver.faces[fname]
        data = plot_data[i]
        
        # Color mapping
        rgba = cmap(norm(data))
        
        # Surface plot
        ax.plot_surface(fg.X, fg.Y, fg.Z, rstride=1, cstride=1,
                       facecolors=rgba,
                       linewidth=0, antialiased=False, shade=False, 
                       alpha=alpha)
    
    # Colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(plot_data)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label("Phi")
    
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.show()

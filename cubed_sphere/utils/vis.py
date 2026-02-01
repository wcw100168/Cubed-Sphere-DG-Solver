import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Any

# Define custom colormap
COLORS_LIST = [
    (0.0, "green"),
    (0.5, "yellow"),
    (1.0, "red"),
]
GREEN_RED_CMAP = LinearSegmentedColormap.from_list("green_red", COLORS_LIST)

def plot_cubed_sphere_state(
    solver: Any, 
    state: np.ndarray, 
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
        The state array (6, N, N).
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
    
    if vmin is None: vmin = np.min(state)
    if vmax is None: vmax = np.max(state)
    
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot each face
    for i, fname in enumerate(solver.topology.FACE_MAP):
        fg = solver.faces[fname]
        data = state[i]
        
        # Color mapping
        rgba = cmap(norm(data))
        
        # Surface plot
        ax.plot_surface(fg.X, fg.Y, fg.Z, rstride=1, cstride=1,
                       facecolors=rgba,
                       linewidth=0, antialiased=False, shade=False, 
                       alpha=alpha)
    
    # Colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(state)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label("Phi")
    
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.show()

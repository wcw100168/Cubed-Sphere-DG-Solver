import numpy as np
from numpy.polynomial.legendre import Legendre
from typing import Tuple

def lg_nodes_weights(a: float, b: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Legendre-Gauss (LG) nodes and weights on interval [a, b].
    
    Parameters
    ----------
    a, b : float
        Interval endpoints.
    N : int
        Number of quadrature nodes.
    """
    xi, wi = np.polynomial.legendre.leggauss(N)
    c = 0.5 * (b - a)
    m = 0.5 * (b + a)
    # Linear map from [-1, 1] to [a, b]
    nodes = c * xi + m
    weights = c * wi
    return nodes, weights

def lgl_nodes_weights(a: float, b: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legendre-Gauss-Lobatto (LGL) nodes and weights on [a, b].
    Nodes includes endpoints.
    """
    if N < 2:
        raise ValueError("LGL requires N >= 2")

    # Use Legendre polynomial of degree N-1, denoted as P_{N-1}
    P = Legendre.basis(N - 1)
    dP = P.deriv()

    # Interior nodes are roots of P'_{N-1}
    # Note: dP.roots() returns sorted roots by default in recent numpy versions,
    # but explicit sort ensures safety.
    x_int = np.sort(dP.roots()) if N > 2 else np.array([], dtype=float)

    # Concatenate endpoints -1 and 1
    x = np.concatenate(([-1.0], x_int, [1.0]))

    # Compute weights on reference interval [-1, 1]
    # Formula: w_j = 2 / (N * (N-1) * [P_{N-1}(x_j)]^2)
    PN1_vals = P(x)
    w = 2.0 / (N * (N - 1) * (PN1_vals ** 2))

    # Linear map to [a, b]
    c = 0.5 * (b - a)
    m = 0.5 * (b + a)
    nodes = c * x + m
    weights = c * w
    return nodes, weights

def lgl_diff_matrix(N: int) -> np.ndarray:
    """
    Construct the 1D LGL differentiation matrix D on [-1, 1].
    
    The metric terms (Jacobian of the mapping) should be applied separately
    when solving on [a, b].
    """
    # 1. Get nodes on reference interval [-1, 1]
    x, _ = lgl_nodes_weights(-1.0, 1.0, N)
    
    # 2. Evaluate P_{N-1}(x) at these nodes
    Pn = Legendre.basis(N - 1)
    Pn_x = Pn(x)
    
    # 3. Compute Distance Matrix (x_i - x_j) using broadcasting
    # xi_grid: columns are identical (x_0, x_0, ... x_0)
    # xj_grid: rows are identical (x_0, x_1, ... x_N)
    xi_grid = x[:, np.newaxis]
    xj_grid = x[np.newaxis, :]
    delta_x = xi_grid - xj_grid
    
    # Avoid division by zero on diagonal by setting diagonal delta_x to 1 (temporary)
    np.fill_diagonal(delta_x, 1.0)
    
    # 4. Compute Off-diagonal entries
    # Formula: D_ij = (P_{N-1}(x_i) / P_{N-1}(x_j)) * (1 / (x_i - x_j))
    # We use outer product for Pn ratios: Pn_x[i] / Pn_x[j]
    Pn_ratio = Pn_x[:, np.newaxis] / Pn_x[np.newaxis, :]
    
    D = Pn_ratio / delta_x
    
    # 5. Correct the Diagonal using Negative Sum Trick
    # (Rows sum to zero for consistency)
    np.fill_diagonal(D, 0.0) # Clear temp values
    row_sum = np.sum(D, axis=1)
    np.fill_diagonal(D, -row_sum)
    
    return D
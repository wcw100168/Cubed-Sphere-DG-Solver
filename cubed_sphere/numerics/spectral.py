import numpy as np
from numpy.polynomial.legendre import Legendre
from typing import Tuple

def lg_nodes_weights(a: float, b: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Legendre-Gauss (LG) nodes and weights on interval [a, b].

    Parameters
    ----------
    a : float
        Start of the interval.
    b : float
        End of the interval.
    N : int
        Number of quadrature nodes.

    Returns
    -------
    nodes : np.ndarray
        Array of N nodes.
    weights : np.ndarray
        Array of N weights.
    """
    xi, wi = np.polynomial.legendre.leggauss(N)  # on [-1,1]
    c = 0.5 * (b - a)
    m = 0.5 * (b + a)
    nodes = c * xi + m
    weights = c * wi
    return nodes, weights

def lgl_nodes_weights(a: float, b: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legendre-Gauss-Lobatto (LGL) nodes and weights on [a, b].
    Using P_{N-1}(x) and its derivative for interior nodes.

    Parameters
    ----------
    a : float
        Start of the interval.
    b : float
        End of the interval.
    N : int
        Number of nodes (must be >= 2).

    Returns
    -------
    nodes : np.ndarray
        Array of N LGL nodes.
    weights : np.ndarray
        Array of N LGL weights.
    
    Raises
    ------
    ValueError
        If N < 2.
    """
    if N < 2:
        raise ValueError("LGL requires N >= 2")

    # Legendre polynomial of degree N-1
    P = Legendre.basis(N - 1)   # P_{N-1}(x) on [-1,1]
    dP = P.deriv()

    # Interior roots of dP (if N>2)
    x_int = np.sort(dP.roots()) if N > 2 else np.array([], dtype=float)

    # Concatenate endpoints +-1
    x = np.concatenate(([-1.0], x_int, [1.0]))

    # Weights on [-1,1]
    PN1_vals = P(x)              # P_{N-1}(x_i)
    w = 2.0 / (N * (N - 1) * (PN1_vals ** 2))

    # Map to [a,b]
    c = 0.5 * (b - a)
    m = 0.5 * (b + a)
    nodes = c * x + m
    weights = c * w
    return nodes, weights


def lgl_diff_matrix(N: int) -> np.ndarray:
    """
    Construct the 1D LGL differentiation matrix D on given N (mapped to [-1, 1]).

    Parameters
    ----------
    N : int
        Number of LGL nodes.

    Returns
    -------
    D : np.ndarray
        The (N, N) differentiation matrix.
    """
    Pn = Legendre.basis(N-1)
    # We compute nodes on [-1, 1] for the standardized Diff Matrix
    x, _ = lgl_nodes_weights(-1.0, 1.0, N)
    
    # Evaluate P_N(x) at standard nodes
    # Note: The code in notebook used Pn = Legendre.basis(N-1), which is P_{N-1}. 
    # Let's verify standard LGL Diff Matrix construction. 
    # Usually involves P_{N-1}(x_i) or P_N(x_i). 
    # The logic in notebook:
    # denominator = (x[i] - x[j]) * Pn_x[j] -> which uses P_{N-1}(x_j).
    # This aligns with common spectral element formulations using L_i(x) constructed from P_{N-1}.
    
    Pn_x = Pn(x)
    
    # Initialize derivative matrix
    D = np.zeros((N, N))
    
    # 1. Off-diagonal elements
    for i in range(N):
        for j in range(N):
            if i != j:
                # Formula: (P_{N-1}(xi) / P_{N-1}(xj)) * (1 / (xi - xj))
                denominator = (x[i] - x[j]) * Pn_x[j]
                if np.abs(denominator) < 1e-15:
                    # Theoretically LGL nodes are unique, so this shouldn't happen unless precision issue
                    D[i, j] = 0.0 
                else:
                    D[i, j] = Pn_x[i] / denominator
    
    # 2. Diagonal elements using Negative Sum Trick
    for i in range(N):
        D[i, i] = -np.sum(D[i, :])
        
    return D

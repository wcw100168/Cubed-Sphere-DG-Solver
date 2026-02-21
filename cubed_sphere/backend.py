import warnings
from typing import Any, Union
import numpy as np

# Type alias for array-like objects
Array = Union[np.ndarray, Any]

def get_backend(name: str):
    """
    Returns the array module (numpy or jax.numpy) based on the name.
    """
    if name == 'numpy':
        return np
    elif name == 'jax':
        try:
            import jax
            import jax.numpy as jnp
            # Scientific computing requires 64-bit precision, but Metal (Apple Silicon) support is limited.
            # We try to enable it, but if it fails, the user might need to rely on f32.
            # For this environment (Metal detected in logs), we will DISABLE x64 to ensure it runs.
            # jax.config.update("jax_enable_x64", True) 
            return jnp
        except ImportError:
            warnings.warn("JAX not installed. Falling back to NumPy.")
            return np
    else:
        raise ValueError(f"Unknown backend: {name}")

def to_backend(arr: Any, backend_module: Any) -> Array:
    """
    Convert an array to the target backend format.

    Parameters
    ----------
    arr : Any
        Input array-like object.
    backend_module : module
        Either numpy or jax.numpy; determines the output type.

    Returns
    -------
    Array
        np.ndarray when backend is numpy; jax.numpy.DeviceArray otherwise.
    """
    if backend_module.__name__ == 'jax.numpy':
        import jax.numpy as jnp
        return jnp.array(arr)
    else:
        return np.array(arr)

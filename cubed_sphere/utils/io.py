import numpy as np
import netCDF4 as nc
import os
from typing import Optional, List, Any

class NetCDFMonitor:
    """
    Monitor that saves the simulation state to a NetCDF file at specified intervals.
    Follows (time, var, face, xi, eta) convention.
    """
    def __init__(self, filename: str, save_interval: float):
        self.filename = filename
        self.save_interval = save_interval
        self.last_save_time = -float('inf')
        self.initialized = False
        
        # Remove existing file to avoid corruption
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def __call__(self, t: float, state: np.ndarray):
        """
        Callback function to be called by the solver.
        state shape: (n_vars, 6, N, N) or (6, N, N)
        """
        # JAX to NumPy conversion
        if not isinstance(state, np.ndarray):
            try:
                state = np.array(state)
            except:
                pass # Try proceeding, maybe it's list-like

        # Check interval
        if t - self.last_save_time < self.save_interval and not np.isclose(t, 0.0):
            # Always save t=0
            if self.initialized: # If it's not the first step and too soon, skip
                return

        self._save_frame(t, state)
        self.last_save_time = t

    def _save_frame(self, t: float, state: np.ndarray):
        # Normalize Shape
        if state.ndim == 3:
            # (6, N, N) -> (1, 6, N, N)
            state_reshaped = state[np.newaxis, ...]
        elif state.ndim == 4:
            state_reshaped = state
        else:
            raise ValueError(f"Unexpected state shape: {state.shape}")

        n_vars, n_faces, N, _ = state_reshaped.shape
        
        mode = 'a' if self.initialized else 'w'
        
        try:
            with nc.Dataset(self.filename, mode) as ds:
                if not self.initialized:
                    # Create dimensions
                    ds.createDimension('time', None) # Unlimited
                    ds.createDimension('var', n_vars)
                    ds.createDimension('face', n_faces)
                    ds.createDimension('xi', N)
                    ds.createDimension('eta', N)
                    
                    # Create variables
                    t_var = ds.createVariable('time', 'f8', ('time',))
                    t_var.units = 'seconds'
                    
                    state_var = ds.createVariable('state', 'f8', ('time', 'var', 'face', 'xi', 'eta'))
                    self.initialized = True
                
                # Write data
                idx = ds.variables['time'].shape[0]
                ds.variables['time'][idx] = t
                ds.variables['state'][idx, :, :, :, :] = state_reshaped
        except Exception as e:
            print(f"Error writing to NetCDF: {e}")


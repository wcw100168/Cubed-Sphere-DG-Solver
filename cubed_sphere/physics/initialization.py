"""
Initialization logic for Cubed Sphere Shallow Water Equations.
Shared between NumPy and JAX solvers to ensure consistency.
"""

import numpy as np
from typing import Dict, Any, Union, Tuple

# Import geometry for type hinting if possible, or just use Dict
# from cubed_sphere.geometry.grid import FaceGrid

def get_initial_state(
    config: Dict[str, Any], 
    faces: Dict[str, Any], 
    case_type: str = "case2",
    **kwargs
) -> np.ndarray:
    """
    Generate the initial state for the SWE solver.
    
    Args:
        config: Configuration dictionary containing 'N', 'R', 'H_avg', 'gravity', 'Omega'.
        faces: Dictionary of FaceGrid objects (from topology.faces).
        case_type: Type of test case ('case2', 'case5', 'case6', 'gaussian').
        **kwargs: Additional parameters (e.g., 'alpha' for rotation angle).
        
    Returns:
        np.ndarray: State array of shape (3, 6, N+1, N+1) where the face axis
                    follows CubedSphereTopology.FACE_MAP. Variables are
                    [Mass (h*sqrt_g), u1_cov, u2_cov].
    """
    # 1. Configuration Constants
    N = config.get('N', 32)
    R = config.get('R', 6.37122e6)
    # H_avg might be passed as h0 or H_avg, check consistency with caller
    h0_avg = config.get('H_avg', 10000.0)
    g = config.get('gravity', 9.80616)
    Omega = config.get('Omega', 7.292e-5)
    
    # Initialize State Tensor (Var, Face, Xi, Eta)
    state = np.zeros((3, 6, N+1, N+1))
    
    FACE_MAP = ["P1", "P2", "P3", "P4", "P5", "P6"]

    # -------------------------------------------------------------------------
    # Case 2: Williamson Steady State Zonal Flow
    # -------------------------------------------------------------------------
    if case_type == "case2":
        alpha_deg = kwargs.get('alpha', 0.0)
        alpha = np.deg2rad(alpha_deg)
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)
        
        # Velocity parameter u0 ~ 38 m/s
        u0_vel = 2.0 * np.pi * R / (12.0 * 24.0 * 3600.0)

        for i, fname in enumerate(FACE_MAP):
            fg = faces[fname]
            lam, th = fg.lon, fg.lat
            
            # --- 1. Rotated Latitude sin(theta') ---
            # sin(th') = sin(th)cos(a) - cos(lam)cos(th)sin(a)
            sin_th_prime = np.sin(th) * cos_alpha - np.cos(lam) * np.cos(th) * sin_alpha
            sin_th = np.sin(th)
            cos_th = np.cos(th)
            sin_lam = np.sin(lam)
            cos_lam = np.cos(lam)
            
            # --- 2. Height Field (h) ---
            # h = h0 - (1/g)(R*Om*u0 + u0^2/2) * sin^2(th')
            term = (R * Omega * u0_vel + 0.5 * u0_vel**2)
            h = h0_avg - (term * sin_th_prime**2) / g
            
            # Store Mass check: h * sqrt_g
            state[0, i] = h * fg.sqrt_g
            
            # --- 3. Velocity Vector (u_s, v_s) in Spherical ---
            # u_s = u0(cos(th)cos(a) + cos(lam)sin(th)sin(a))
            # v_s = -u0 sin(lam)sin(a)
            u_sph = u0_vel * (cos_th * cos_alpha + cos_lam * sin_th * sin_alpha)
            v_sph = -u0_vel * (-np.cos(lam) * np.sin(th) * sin_alpha) # Original was -u0*sin(lam)*sin(a) which assumes alpha=0 logic?
            # Wait, formula 2.2 in Williamson:
            # v_s = -u0 * sin(lam_prime) * sin(theta_prime)? No.
            # v_s = -u0 * sin(lam) * sin(alpha) (Eq 2.2)
            # My previous code: v_sph = -u0_vel * np.sin(lam) * sin_alpha
            v_sph = -u0_vel * sin_lam * sin_alpha
            
            # --- 4. Project to Cartesian (Vx, Vy, Vz) ---
            # e_lam = (-sin(lam), cos(lam), 0)
            # e_th  = (-sin(th)cos(lam), -sin(th)sin(lam), cos(th))
            e_lam_x, e_lam_y, e_lam_z = -sin_lam, cos_lam, 0.0
            e_th_x, e_th_y, e_th_z    = -sin_th*cos_lam, -sin_th*sin_lam, cos_th
            
            Vx = u_sph * e_lam_x + v_sph * e_th_x
            Vy = u_sph * e_lam_y + v_sph * e_th_y
            Vz = u_sph * e_lam_z + v_sph * e_th_z
            
            # --- 5. Project to Covariant Basis (u_1, u_2) ---
            # u_i = V . g_i
            # g_i vectors from FaceGrid
            b1 = Vx*fg.g1_vec[...,0] + Vy*fg.g1_vec[...,1] + Vz*fg.g1_vec[...,2]
            b2 = Vx*fg.g2_vec[...,0] + Vy*fg.g2_vec[...,1] + Vz*fg.g2_vec[...,2]
            
            state[1, i] = b1
            state[2, i] = b2

    # -------------------------------------------------------------------------
    # Case 6: Rossby-Haurwitz Wave (Wavenumber 4)
    # -------------------------------------------------------------------------
    elif case_type == "case6":
        omega = 7.848e-6 # K=omega
        K = 7.848e-6
        R_wave = 4.0
        h0_c6 = 8000.0 # Base height for Case 6
        
        for i, fname in enumerate(FACE_MAP):
            fg = faces[fname]
            lam, th = fg.lon, fg.lat
            sin_lat, cos_lat = np.sin(th), np.cos(th)
            cos_Rlam = np.cos(R_wave * lam)
            cos_2Rlam = np.cos(2.0 * R_wave * lam)
            sin_Rlam = np.sin(R_wave * lam)
            
            # --- Height Field Terms (A, B, C) ---
            # A(theta)
            t1 = 0.5 * omega * (2.0 * Omega + omega) * (cos_lat**2)
            c2R = cos_lat**(2*R_wave)
            c2R_plus2 = cos_lat**(2*R_wave + 2)
            c2R_minus2 = cos_lat**(2*R_wave - 2)
            
            t2 = 0.25 * K**2 * (
                (R_wave + 1) * c2R_plus2 + 
                (2 * R_wave**2 - R_wave - 2) * c2R - 
                2 * R_wave**2 * c2R_minus2
            )
            A = t1 + t2
            
            # B(theta)
            b_num = 2.0 * (Omega + omega) * K
            b_den = (R_wave + 1) * (R_wave + 2)
            b_brack = ( (R_wave**2 + 2*R_wave + 2) - (R_wave+1)**2 * cos_lat**2 )
            B = (b_num / b_den) * (cos_lat**R_wave) * b_brack
            
            # C(theta)
            c_brack = (R_wave + 1) * cos_lat**2 - (R_wave + 2)
            C = 0.25 * K**2 * c2R * c_brack
            
            # Total Height h
            gh = g * h0_c6 + (R**2) * (A + B * cos_Rlam + C * cos_2Rlam)
            h = gh / g
            
            state[0, i] = h * fg.sqrt_g
            
            # --- Velocity Field ---
            u_term1 = R * omega * cos_lat
            u_term2 = R * K * (cos_lat**(R_wave-1)) * (R_wave * sin_lat**2 - cos_lat**2) * cos_Rlam
            u_sph = u_term1 + u_term2
            
            v_sph = -R * K * R_wave * (cos_lat**(R_wave-1)) * sin_lat * sin_Rlam
            
            # Project to Cartesian -> Covariant
            sin_lam, cos_lam = np.sin(lam), np.cos(lam)
            
            Vx = u_sph * (-sin_lam) + v_sph * (-sin_lat * cos_lam)
            Vy = u_sph * (cos_lam)  + v_sph * (-sin_lat * sin_lam)
            Vz = u_sph * (0.0)      + v_sph * (cos_lat)
            
            state[1, i] = Vx*fg.g1_vec[...,0] + Vy*fg.g1_vec[...,1] + Vz*fg.g1_vec[...,2]
            state[2, i] = Vx*fg.g2_vec[...,0] + Vy*fg.g2_vec[...,1] + Vz*fg.g2_vec[...,2]
            
    # -------------------------------------------------------------------------
    # Gaussian Bump (Local Perturbation)
    # -------------------------------------------------------------------------
    elif case_type == "gaussian":
        h_pert = kwargs.get('h_pert', 100.0) # 100m perturbation
        center_lon = kwargs.get('lon0', 0.0)
        center_lat = kwargs.get('lat0', 0.0)
        width = kwargs.get('width', 0.1) # Radians? Or normalized? Usually radians.
        
        # Base flow usually rest or solid body. Let's assume rest for now.
        
        for i, fname in enumerate(FACE_MAP):
            fg = faces[fname]
            lam, th = fg.lon, fg.lat
            
            # Great Circle Distance 'r' from center
            # cos(r) = sin(lat)sin(lat0) + cos(lat)cos(lat0)cos(lon-lon0)
            arg = np.sin(th)*np.sin(center_lat) + np.cos(th)*np.cos(center_lat)*np.cos(lam - center_lon)
            # Clip to [-1, 1] for acos safety
            arg = np.clip(arg, -1.0, 1.0)
            dist = np.arccos(arg)
            
            h = h0_avg + h_pert * np.exp(-(dist**2) / (width**2))
            
            state[0, i] = h * fg.sqrt_g
            # Zero velocity
            state[1, i] = 0.0
            state[2, i] = 0.0

    else:
        # Default / Fallback (Rest)
        print(f"Warning: Unknown case type '{case_type}'. Defaulting to rest state.")
        for i, fname in enumerate(FACE_MAP):
            fg = faces[fname]
            state[0, i] = h0_avg * fg.sqrt_g
            state[1, i] = 0.0
            state[2, i] = 0.0
            
    return state

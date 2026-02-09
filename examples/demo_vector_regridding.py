
import numpy as np
import matplotlib.pyplot as plt
from cubed_sphere.geometry.grid import CubedSphereEquiangular, CubedSphereTopology
from cubed_sphere.utils import regrid

def verify_wind_field():
    print("\n--- Running Vector Field Regridding Demo ---")
    """
    Demo for vector field regridding from Lat-Lon grid to Cubed Sphere.
    Verifies metric tensor transformations.
    """
   
    # 1. Setup Ground Truth: Solid Body Rotation (Zonal Wind)
    # u_east = 40.0 m/s, v_north = 0.0
    u_ref = 40.0
    R = 6371000.0 # Earth Radius in meters roughly, or just 1.0 for unit sphere
    # Let's use R=1.0 for simplicity unless we want physical units. 
    # The prompt says u_east=40.0. If R=1.0, 40 is huge angular velocity. 
    # Let's assume consistent units. If R=1.0, u=40 means 40 radians/sec?
    # Let's stick to R=1.0 and u=1.0 to be safe, OR use R=6.371e6 and u=40.
    # The user prompt specifically says u_east = 40.0. Let's use that.
    
    # 1.1 Source Grid (Lat/Lon)
    n_lat, n_lon = 90, 180
    lat_deg = np.linspace(-90, 90, n_lat)
    lon_deg = np.linspace(-180, 180, n_lon)
    
    # Create 2D arrays for u and v
    # shape (n_lat, n_lon)
    u_src = np.full((n_lat, n_lon), u_ref)
    v_src = np.full((n_lat, n_lon), 0.0)
    
    # 2. Setup Target: Cubed Sphere
    N = 32
    cs_geo = CubedSphereEquiangular(R=1.0) # Using Unit Sphere for geometry calcs? 
    # If we interpolate u=40 onto R=1 sphere, the physical magnitude is just a number.
    # BUT, compute_contravariant_vel uses R.
    # If source u is physical velocity (m/s), and R=1m, then angular velocity is 40 rad/s.
    # We should match R if we want physical meaning, or just ignore units.
    # Let's set R=1.0 for the test geometry to match default.
    
    # 3. Perform Regridding
    print("Regridding Vector Field...")
    
    # Store results
    u1_all = {}
    u2_all = {}
    faces = {}
    
    topology = CubedSphereTopology()
    for fname in topology.FACE_MAP:
        fg = cs_geo.generate_face(N, fname)
        faces[fname] = fg
        # Pass geometry_handler to use correct R and logic
        u1, u2 = regrid.from_latlon_vector(fg, lat_deg, lon_deg, u_src, v_src, geometry_handler=cs_geo)
        u1_all[fname] = u1
        u2_all[fname] = u2

    # 4. Verification Logic
    print("\n--- Verification Metrics ---")
    
    max_error_rel = 0.0
    
    for fname in topology.FACE_MAP:
        fg = faces[fname]
        u1 = u1_all[fname]
        u2 = u2_all[fname]
        
        # Metric 1: Physical Magnitude Check
        # |V|^2 = g11*u1^2 + 2*g12*u1*u2 + g22*u2^2
        # However, we only have u1, u2 (contravariant) and sqrt_g (Jacobian).
        # We don't have the metric tensor components g_ij explicitly stored in FaceGrid yet.
        # But we know u1, u2 are computed from u_sph, v_sph in `geometry/grid.py`.
        # And u_sph = V . e_lambda, v_sph = V . e_theta.
        # Magnitude^2 = u_sph^2 + v_sph^2 (since spherical basis is orthonormal).
        # Wait, local basis (u1, u2) is NOT orthonormal.
        # We can reconstruct Spherical components to check magnitude!
        # Or better: We can compute the Cartesian magnitude from u1, u2?
        # Actually easier: The regridding internally computed u_sph, v_sph.
        # But we only returned u1, u2.
        # To verify u1, u2 are correct, we should convert them BACK to physical magnitude using metric.
        
        # Let's Compute Metric Tensor g_ij on the fly for verification.
        # Or... rely on the fact that we can invert the relation?
        # u1 = ... u_sph ...
        # It's cleaner to verify |V| approx 40.0.
        
        # Let's derive u_sph_reconstructed from u1, u2? 
        # [u1] = A^-1 [u_s]
        # [u2]        [v_s]
        # So [u_s] = A [u1]
        #    [v_s]     [u2]
        # We can copy the A matrix logic from grid.py or just trust the interpolation intermediate?
        # No, we must verify the OUTPUT u1, u2.
        
        # We'll re-implement the A matrix logic here for verification (or import if we could refactor grid.py)
        # For now, let's just use the `geometry.compute_contravariant_vel` logic in reverse.
        # Actually, let's look at `grid.py`:
        # A matrix transforms (u1, u2) -> something?
        # No, grid.py says: u1, u2 = inv_det * ( ... u_sphere ... )
        # implies [u1, u2]^T = A^-1 [u_s, v_s]^T
        # Thus [u_s, v_s]^T = A [u1, u2]^T
        
        # Let's reconstruct u_s, v_s using A matrix (copy-paste logic basically due to encapsulation)
        # This is a good independent check.
        
        lam, theta = cs_geo.lonlat_from_xyz(fg.X, fg.Y, fg.Z)
        sec2_a = 1.0 / (np.cos(fg.alpha) ** 2)
        sec2_b = 1.0 / (np.cos(fg.beta) ** 2)
        c_lam, s_lam = np.cos(lam), np.sin(lam)
        c_th, s_th = np.cos(theta), np.sin(theta)
        
        A = np.zeros(fg.alpha.shape + (2, 2))
        if fg.name == "P1":
            A[...,0,0], A[...,0,1] = cs_geo.R * c_lam**2 * c_th * sec2_a, 0.0
            A[...,1,0], A[...,1,1] = -cs_geo.R * s_lam * c_lam * s_th * c_th * sec2_a, cs_geo.R * c_lam * c_th**2 * sec2_b
        elif fg.name == "P2":
            A[...,0,0], A[...,0,1] = cs_geo.R * s_lam**2 * c_th * sec2_a, 0.0
            A[...,1,0], A[...,1,1] = cs_geo.R * s_lam * c_lam * s_th * c_th * sec2_a, cs_geo.R * s_lam * c_th**2 * sec2_b
        elif fg.name == "P3":
             A[...,0,0], A[...,0,1] = cs_geo.R * c_lam**2 * c_th * sec2_a, 0.0
             A[...,1,0], A[...,1,1] = -cs_geo.R * s_lam * c_lam * s_th * c_th * sec2_a, -cs_geo.R * c_lam * c_th**2 * sec2_b
        elif fg.name == "P4":
             A[...,0,0], A[...,0,1] = cs_geo.R * s_lam**2 * c_th * sec2_a, 0.0
             A[...,1,0], A[...,1,1] = cs_geo.R * s_lam * c_lam * s_th * c_th * sec2_a, -cs_geo.R * s_lam * c_th**2 * sec2_b
        elif fg.name == "P5": # Top
             A[...,0,0], A[...,0,1] = cs_geo.R * c_lam * s_th * sec2_a, cs_geo.R * s_lam * s_th * sec2_b
             A[...,1,0], A[...,1,1] = -cs_geo.R * s_lam * s_th**2 * sec2_a, cs_geo.R * c_lam * s_th**2 * sec2_b
        elif fg.name == "P6": # Bottom
             A[...,0,0], A[...,0,1] = -cs_geo.R * c_lam * s_th * sec2_a, cs_geo.R * s_lam * s_th * sec2_b
             A[...,1,0], A[...,1,1] = cs_geo.R * s_lam * s_th**2 * sec2_a, cs_geo.R * c_lam * s_th**2 * sec2_b
             
        # Reconstruct Spherical Velocity
        # [u_s] = [A00 A01] [u1]
        # [v_s]   [A10 A11] [u2]
        
        u_s_rec = A[...,0,0]*u1 + A[...,0,1]*u2
        v_s_rec = A[...,1,0]*u1 + A[...,1,1]*u2
        
        mag_rec = np.sqrt(u_s_rec**2 + v_s_rec**2)
        
        # Error
        err = np.abs(mag_rec - u_ref) / u_ref
        max_err = np.max(err)
        max_error_rel = max(max_error_rel, max_err)
        
        print(f"Face {fname}: Max Rel Error = {max_err*100:.4f}%")
        
    print(f"\nOverall Max Relative Interpolation Error: {max_error_rel*100:.4f}%")
    
    # Metric 2: Direction Check on P5 (North Pole)
    # At the North Pole, Zonal flow should look like flow across the face?
    # Actually, flow is "East".
    # At North Pole (Face 5 center roughly), "East" is not defined well?
    # P5 center is North Pole.
    # At exact North Pole, East is undefined.
    # But for a zonal flow defined as u_lon=Constant, near the pole it spins around.
    # Wait, u_ref=Constant means constant angular velocity or linear?
    # If u_lon = 40 m/s everywhere, and radius decreases near pole, angular velocity shoots up?
    # No, u_lon IS the linear velocity component in East direction.
    # If u_lon=40 everywhere, then at the pole it's a singularity direction-wise but magnitude is 40.
    # The vectors form a swirl around the pole? No, they are tangent to latitude circles.
    # Latitude circles get small. The vectors loop around tightly.
    
    # Let's inspect P5 (Top Face).
    # The wind is "East".
    # P5 is viewed from top. +Y is direction of what?
    # Let's visualize with quiver 
    
    try:
        # Plotting Face 5
        u1_p5 = u1_all["P5"]
        u2_p5 = u2_all["P5"]
        
        # We really want to plot u_s_rec, v_s_rec or similar to see direction or x,y on geometric map
        # Better: Plot Reconstructed u_s_rec, v_s_rec on P1 (Equator) and P5 (Pole)
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: P1 Equator
        plt.subplot(1, 2, 1)
        # On P1, East is roughly +x direction?
        # Check P1: Center is (1,0,0) -> Equator, Lon=0.
        # East is +Y direction (0,1,0).
        # North is +Z direction (0,0,1).
        # u1 corresponds roughly to d/dxi ~ d/dy?
        # P1 mapping: X = a, Y = x, Z = y (Wait, check grid.py mapping)
        # grid.py: P1: X=s*a, Y=s*x, Z=s*y. 
        # So x is Y-like (East), y is Z-like (North).
        # So u1 should be ~positive, u2 should be ~0.
        u1_p1 = u1_all["P1"]
        u2_p1 = u2_all["P1"]
        plt.quiver(u1_p1[::4, ::4], u2_p1[::4, ::4])
        plt.title("Face 1 (Equator) - Contravariant Flow\nShould be mainly Horizontal (East)")
        
        # Subplot 2: P5 North Pole
        plt.subplot(1, 2, 2)
        # P5: X = -y, Y = x, Z = a (Top).
        # Center is (0,0,1) -> North Pole.
        # Flow is Zonal (Eastward).
        # East vector is [-sin(lam), cos(lam), 0].
        # P5 x,y coords map to global like: x->Y, -y->X ?
        # It's a swirl.
        plt.quiver(u1_p5[::4, ::4], u2_p5[::4, ::4])
        plt.title("Face 5 (North Pole) - Contravariant Flow\nShould be swirling?")
        
        plt.savefig("verify_wind_field.png")
        print("Saved quiver plot to verify_wind_field.png")
    
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    verify_wind_field()

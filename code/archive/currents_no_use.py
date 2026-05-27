# Archive of unused functions from magnetic_field.py
# These functions are deprecated and replaced by optimized versions
# DO NOT USE in active code - kept for reference only

import numpy as np
from scipy.special import ellipk, ellipe

mu0 = 4 * np.pi * 1e-7

def loop_segments(R=1.0, N=512):
    """
    DEPRECATED: Old discretization method for current loop.
    Replaced by B_analytical approach in b_s_l_optimized.
    
    Divides a circular loop into N segments.
    
    Args:
        R: Radius of the loop
        N: Number of segments
    
    Returns:
        rp: (N, 3) array of segment positions
        dl: (N, 3) array of segment direction vectors
    """
    phi = np.linspace(0, 2*np.pi, N, endpoint=False) 
    dphi = 2*np.pi / N 
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = np.zeros_like(x)
    rp = np.stack((x, y, z), axis=1) 
    dl = np.stack((-R * np.sin(phi), R * np.cos(phi), np.zeros_like(phi)), axis=1) * dphi
    return rp, dl


def bio_savar_loop_point(r_obs, R=1.0, I=1.0, N=1024):
    """
    DEPRECATED: Biot-Savart law for a single point observation.
    Replaced by B_analytical in magnetic_field.py.
    
    Calculates magnetic field at a single observation point for a circular loop.
    Uses numerical integration via segment discretization.
    
    Args:
        r_obs: (3,) observation point coordinates
        R: Loop radius
        I: Current through loop
        N: Number of segments for discretization
    
    Returns:
        B: (3,) magnetic field vector at observation point
    """
    rp, dl = loop_segments(R, N)
    Rvecs = r_obs.reshape(1,3) - rp
    norms = np.linalg.norm(Rvecs, axis=1)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    B = (mu0 / (4*np.pi)) * I * np.sum(np.cross(dl, Rvecs) / (norms**3).reshape(-1,1), axis=0)
    return B


def b_s_l(obs_points, I_matrix, N_seg, n, m, all_coordinates, normals):
    """
    DEPRECATED: Old implementation of magnetic field calculation.
    Replaced by b_s_l_optimized which uses analytical formula.
    
    This was a segment-based numerical integration approach.
    The new method (b_s_l_optimized) is much faster and more accurate.
    
    Args:
        obs_points: (N_obs, 3) observation points
        I_matrix: (m, n) current matrix
        N_seg: Number of segments per loop
        n: Number of rings per stack
        m: Number of stacks
        all_coordinates: Loop segment coordinates
        normals: Normal vectors to loop planes
    
    Returns:
        B_total: (N_obs, 3) magnetic field vectors
    """
    N_obs = len(obs_points)
    B_total = np.zeros((N_obs, 3), dtype=complex)
    
    dtheta = 2 * np.pi / N_seg 
    
    for stack in range(m):
        normal = normals[stack]
        
        for ring in range(n):
            ring_glob = stack * n + ring
            seg_start = ring_glob * N_seg
            seg_end = seg_start + N_seg
            ring_points = all_coordinates[seg_start:seg_end]
            
            current = I_matrix[stack, ring]
            ring_center = np.mean(ring_points, axis=0)
            
            
            rad = ring_points - ring_center
            R = np.mean(np.linalg.norm(rad, axis=1))
            

            dl = np.cross(normal, rad)
            dl_norms = np.linalg.norm(dl, axis=1, keepdims=True)
            dl = dl / dl_norms
            
            dl_length = R * dtheta
            dl_vecs = dl * dl_length
            
            for seg in range(N_seg):
                dl_vec = dl_vecs[seg]
                seg_pos = ring_points[seg]
                
                r_vecs = obs_points - seg_pos
                r_dist = np.linalg.norm(r_vecs, axis=1, keepdims=True)
            
                r_dist = np.where(r_dist < 1e-12, 1e-12, r_dist)
                
                cross = np.cross(dl_vec, r_vecs)
                dB = (mu0 / (4 * np.pi)) * current * cross / (r_dist**3)
                B_total += dB
    
    return B_total


# Legacy parameters (example configuration - do not use
# Use config.py and MRIOptimizer class instead

# fixed_params = {
#     'r_ohm': 3.73e-4,           # Ом
#     'A': 0.08,              # м
#     'U_0': 500,             # В
#     'N': 200,               # кол-во точек
#     'grid': np.linspace(-0.2, 0.2, 200),  # сетка для оценки однородности
#     'L_own': 0.21e-6,          # Гн
#     'R': np.full(n, 0.035),    # м (радиус колец)
#     'n': 5,                 # шт (кол-во колец в стопке)
#     'm': 6,                 # шт (кол-во стопок)
#     'delta': np.array([0.01, 0.001, 0.01, 0.001]),          # м (зазор между кольцами)
#     'B_target': 10e-6,        # Целевая индукция (10 мкТл)
#     'weight_magnitude': 0.3,  # Вес величины поля (0.3 = 30% важности)
#     'B_min_threshold': 1e-6
# }
import time
import numpy as np
from scipy.special import ellipk, ellipe

import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from src.geometry import points_on_rings_general, ring_center_general
from src.currents import Z_self_matrix, generate_voltage_array, calc_I
from src.inductance import inductance_matrix_nonqs

mu0 = 4 * np.pi * 1e-7
mu0_over_4pi = mu0 / (4 * np.pi)


# ---------------------------------------------------------------------------
# B_analytical -- без изменений
# ---------------------------------------------------------------------------
def B_analytical(r_obs, R, I, center, normal):
    r_local = r_obs - center
    z = np.dot(r_local, normal)
    rho_vec = r_local - z * normal
    rho = np.linalg.norm(rho_vec)

    if rho < 1e-12:
        B_z_val = (mu0 * I * R ** 2) / (2 * (R ** 2 + z ** 2) ** 1.5)
        return B_z_val * normal

    rho_unit = rho_vec / rho
    k2 = (4 * R * rho) / ((R + rho) ** 2 + z ** 2)
    denom = np.sqrt((R + rho) ** 2 + z ** 2)
    denom_special = (R - rho) ** 2 + z ** 2
    if denom_special < 1e-18:
        denom_special = 1e-18

    K = ellipk(k2)
    E = ellipe(k2)
    factor = (mu0 * I) / (2 * np.pi)

    B_rho = factor * (z / (rho * denom)) * (
        -K + ((R ** 2 + rho ** 2 + z ** 2) / denom_special) * E
    )
    B_z = factor * (1 / denom) * (
        K + ((R ** 2 - rho ** 2 - z ** 2) / denom_special) * E
    )
    return B_rho * rho_unit + B_z * normal


# ---------------------------------------------------------------------------
# b_s_l_optimized -- с таймером
# ---------------------------------------------------------------------------
def b_s_l_optimized(obs_points, I_matrix, n, m, ring_centers, normals, radii):
    t0 = time.time()

    N_obs = len(obs_points)
    B_total = np.zeros((N_obs, 3), dtype=complex)

    for stack in range(m):
        normal = normals[stack]
        for ring in range(n):
            ring_idx = stack * n + ring
            current = I_matrix[stack, ring]
            center = ring_centers[ring_idx]
            R = radii[ring]

            for i in range(N_obs):
                B_total[i] += B_analytical(obs_points[i], R, current, center, normal)

    dt = time.time() - t0
    return B_total


# ---------------------------------------------------------------------------
# quality_metric_std -- с таймером
# ---------------------------------------------------------------------------
def quality_metric_std(I, n, m, current_centers, current_normals,
                       R_array, R_domain, N_grid=20):

    x = np.linspace(-R_domain, R_domain, N_grid)
    y = np.linspace(-R_domain, R_domain, N_grid)
    X, Y = np.meshgrid(x, y)

    mask = (X ** 2 + Y ** 2) <= R_domain ** 2
    X_inside = X[mask]
    Y_inside = Y[mask]
    Z_inside = np.zeros_like(X_inside)

    obs_points = np.stack((X_inside, Y_inside, Z_inside), axis=1)

    B_complex = b_s_l_optimized(obs_points, I, n, m,
                                 current_centers, current_normals, R_array)
    B_abs = np.linalg.norm(B_complex, axis=1)

    std_B = np.std(B_abs)
    mean_B = np.mean(B_abs)


    cv = std_B / mean_B
    return cv


# ---------------------------------------------------------------------------
# quality_metric_lin -- с таймером
# ---------------------------------------------------------------------------
def quality_metric_lin(I, n, m, current_centers, current_normals,
                       R_array, R_domain, N_grid=40):
    t0 = time.time()

    x = np.linspace(-R_domain, R_domain, N_grid)
    y = np.linspace(-R_domain, R_domain, N_grid)
    X, Y = np.meshgrid(x, y)

    mask = (X ** 2 + Y ** 2) <= R_domain ** 2
    X_inside = X[mask]
    Y_inside = Y[mask]
    Z_inside = np.zeros_like(X_inside)

    obs_points = np.stack((X_inside, Y_inside, Z_inside), axis=1)

    B_complex = b_s_l_optimized(obs_points, I, n, m,
                                 current_centers, current_normals, R_array)
    B_abs = np.linalg.norm(B_complex, axis=1)

    max_B = np.max(B_abs)
    min_B = np.min(B_abs)
    mean_B = np.mean(B_abs)

    dt = time.time() - t0
    if mean_B < 1e-12:
        print(f"[TIME quality_lin] N_grid={N_grid}, N_obs={len(obs_points)}, "
              f"время = {dt * 1000:.2f} мс (mean_B=0)")
        return 1.0

    cv = (max_B - min_B) / mean_B
    print(f"[TIME quality_lin] N_grid={N_grid}, N_obs={len(obs_points)}, "
          f"время = {dt * 1000:.2f} мс, CV={cv:.4f}")
    return cv


# ---------------------------------------------------------------------------
# quality_metric_article -- с таймером
# ---------------------------------------------------------------------------
def quality_metric_article(I, n, m, current_centers, current_normals,
                            R_array, R_domain, N_grid=40):
    t0 = time.time()

    x = np.linspace(-R_domain, R_domain, N_grid)
    y = np.linspace(-R_domain, R_domain, N_grid)
    X, Y = np.meshgrid(x, y)

    mask = (X ** 2 + Y ** 2) <= R_domain ** 2
    X_inside = X[mask]
    Y_inside = Y[mask]
    Z_inside = np.zeros_like(X_inside)

    obs_points = np.stack((X_inside, Y_inside, Z_inside), axis=1)

    B_complex = b_s_l_optimized(obs_points, I, n, m,
                                 current_centers, current_normals, R_array)
    B_abs = np.linalg.norm(B_complex, axis=1)

    max_B = np.max(B_abs)
    min_B = np.min(B_abs)

    dt = time.time() - t0
    if R_domain < 1e-12:
        print(f"[TIME quality_article] N_grid={N_grid}, "
              f"время = {dt * 1000:.2f} мс (R_domain=0)")
        return 1.0

    cv = 2.0 * (max_B - min_B) / (R_domain ** 2)
    print(f"[TIME quality_article] N_grid={N_grid}, "
          f"N_obs={len(obs_points)}, время = {dt * 1000:.2f} мс, CV={cv:.4f}")
    return cv
import numpy as np
from baryrat import aaa
from scipy.signal import find_peaks

from src.currents import Z_self_matrix, generate_voltage_array, calc_I
from src.magnetic_field import b_s_l_optimized

FREQ_MIN_HZ_DEFAULT = 30e6
FREQ_MAX_HZ_DEFAULT = 90e6
N_COARSE_DEFAULT    = 300
N_FINE_DEFAULT      = 8000
AAA_TOL_DEFAULT     = 1e-9
AAA_MMAX_DEFAULT    = 100

def _response_at(f_hz, L, n, m, radii, C_array, r_ohm, U_const,
                 ring_centers, normals, obs_point):
    omega = 2 * np.pi * f_hz
    Z_self = Z_self_matrix(r=r_ohm, C=C_array, n=n, m=m, R=radii, omega=omega)
    I_matrix = calc_I(Z_self=Z_self, U=U_const, omega=omega, L=L, n=n, m=m)

    if I_matrix is None or not np.all(np.isfinite(I_matrix)):
        return np.full(3, np.nan + 0j)

    B_vec = b_s_l_optimized(
        obs_point, I_matrix, n, m, ring_centers, normals, radii
    )
    return B_vec[0, :]


def find_resonance_frequency(
    L, n, m, radii, A, gaps,
    C_value, r_ohm, U_0,
    ring_centers, normals,
    obs_point=None,
    N_seg=128,
    f_min=FREQ_MIN_HZ_DEFAULT,
    f_max=FREQ_MAX_HZ_DEFAULT,
    n_coarse=N_COARSE_DEFAULT,
    n_fine=N_FINE_DEFAULT,
    aaa_tol=AAA_TOL_DEFAULT,
    aaa_mmax=AAA_MMAX_DEFAULT,
    prominence_frac=0.01,
    return_curve=False,
    return_all_peaks=False,
):
    if obs_point is None:
        obs_point = np.array([[0.0, 0.0, 0.0]])

    C_array = np.full(n * m, C_value)
    U_const = generate_voltage_array(U_0=U_0, m=m, n=n)

    freqs_coarse = np.linspace(f_min, f_max, n_coarse)
    B_coarse = np.empty((n_coarse, 3), dtype=complex)
    for k, f in enumerate(freqs_coarse):
        B_coarse[k] = _response_at(f, L, n, m, radii, C_array, r_ohm, U_const, ring_centers, normals, obs_point)

    good = np.all(np.isfinite(B_coarse), axis=1)
    if good.sum() < 4:
        return _empty_return(f_min, freqs_coarse, B_coarse,
                             return_curve, return_all_peaks)

    f0, f1 = f_min, f_max
    s_coarse = (freqs_coarse[good] - f0) / (f1 - f0)
    B_good = B_coarse[good]

    try:
        r_aaa = [
            aaa(s_coarse, B_good[:, comp], tol=aaa_tol, mmax=aaa_mmax)
            for comp in range(3)
        ]
    except Exception:
        return _empty_return(f_min, freqs_coarse, B_coarse,
                             return_curve, return_all_peaks)

    s_fine = np.linspace(0.0, 1.0, n_fine)
    f_fine = f0 + s_fine * (f1 - f0)

    B_fine = np.empty((n_fine, 3), dtype=complex)
    for comp in range(3):
        B_fine[:, comp] = r_aaa[comp](s_fine)

    mag_fine = np.linalg.norm(B_fine, axis=1)

    peaks, _ = find_peaks(
        mag_fine,
        prominence=prominence_frac * mag_fine.max(),
        distance=3,
    )

    if len(peaks) == 0:
        idx = int(np.argmax(mag_fine))
        peaks_freqs = np.array([f_fine[idx]])
        f_res = f_fine[idx]
        mag_at_res = mag_fine[idx]
    else:
        df = f_fine[1] - f_fine[0]
        peaks_freqs = np.empty(len(peaks))
        peaks_vals  = np.empty(len(peaks))
        for i, p in enumerate(peaks):
            if 0 < p < n_fine - 1:
                y0, y1, y2 = mag_fine[p-1], mag_fine[p], mag_fine[p+1]
                denom = y0 - 2*y1 + y2
                if abs(denom) > 0:
                    delta = np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5)
                    peaks_freqs[i] = f_fine[p] + delta * df
                    peaks_vals[i]  = y1 - 0.25 * (y0 - y2) * delta
                else:
                    peaks_freqs[i] = f_fine[p]
                    peaks_vals[i]  = mag_fine[p]
            else:
                peaks_freqs[i] = f_fine[p]
                peaks_vals[i]  = mag_fine[p]

        # берётся пик с максимальной амплитудой
        idx_best = int(np.argmax(peaks_vals))
        f_res = peaks_freqs[idx_best]
        mag_at_res = peaks_vals[idx_best]

    if return_curve and return_all_peaks:
        return f_res, f_fine, B_fine, peaks_freqs
    if return_curve:
        return f_res, f_fine, B_fine
    if return_all_peaks:
        return f_res, peaks_freqs
    return f_res


def _empty_return(f_min, freqs_coarse, B_coarse, return_curve, return_all_peaks):
    empty = np.array([])
    if return_curve and return_all_peaks:
        return f_min, freqs_coarse, B_coarse, empty
    if return_curve:
        return f_min, freqs_coarse, B_coarse
    if return_all_peaks:
        return f_min, empty
    return f_min
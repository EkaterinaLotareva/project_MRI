# src/inductance.py
import numpy as np
from scipy import integrate
import src.config as config

mu_0 = 4 * np.pi * 1e-7
C_LIGHT = 299792458.0


def L_int_par(dx: float, dy: float, dz: float, r1: float, r2: float, k: float) -> complex:
    def internal(phi1, phi2):
        A = dz**2 + dx**2 + dy**2 + 2*r2*(dx*np.cos(phi2) + dy*np.sin(phi2)) + r1**2 + r2**2
        B = -2*dx*r1 - 2*r1*r2*np.cos(phi2)
        C = -2*dy*r1 - 2*r1*r2*np.sin(phi2)
        arg = max(A + B*np.cos(phi1) + C*np.sin(phi1), 1e-14)
        R = np.sqrt(arg)
        spr = np.cos(phi2 - phi1)
        return spr * np.cos(k*R) / R, spr * np.sin(k*R) / R

    outer_re = lambda p2: integrate.quad(lambda p1: internal(p1, p2)[0], 0, 2*np.pi, epsabs=1e-6, epsrel=1e-4, limit=50)[0]
    outer_im = lambda p2: integrate.quad(lambda p1: internal(p1, p2)[1], 0, 2*np.pi, epsabs=1e-6, epsrel=1e-4, limit=50)[0]

    re = integrate.quad(outer_re, 0, 2*np.pi, epsabs=1e-6, epsrel=1e-4, limit=50)[0]
    im = integrate.quad(outer_im, 0, 2*np.pi, epsabs=1e-6, epsrel=1e-4, limit=50)[0]

    # Корректный знак запаздывания e^(-jkR) = cos(kR) - j*sin(kR)
    return (re - 1j*im) * (r1 * r2 * mu_0 / (4 * np.pi))


def L_int_angle(b1: float, b2: float, r1: float, r2: float, alpha: float, k: float) -> complex:
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)

    def internal(phi1, phi2):
        x2 = b2 * cos_a - r2 * sin_a * np.cos(phi2)
        y2 = b2 * sin_a + r2 * cos_a * np.cos(phi2)
        z2 = r2 * np.sin(phi2)
        A = (x2 - b1)**2 + y2**2 + z2**2 + r1**2
        B = -2 * r1 * y2
        C = -2 * r1 * z2
        arg = max(A + B*np.cos(phi1) + C*np.sin(phi1), 1e-14)
        R = np.sqrt(arg)
        spr = cos_a * np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2)
        return spr * np.cos(k*R) / R, spr * np.sin(k*R) / R

    outer_re = lambda p2: integrate.quad(lambda p1: internal(p1, p2)[0], 0, 2*np.pi, epsabs=1e-6, epsrel=1e-4, limit=50)[0]
    outer_im = lambda p2: integrate.quad(lambda p1: internal(p1, p2)[1], 0, 2*np.pi, epsabs=1e-6, epsrel=1e-4, limit=50)[0]

    re = integrate.quad(outer_re, 0, 2*np.pi, epsabs=1e-6, epsrel=1e-4, limit=50)[0]
    im = integrate.quad(outer_im, 0, 2*np.pi, epsabs=1e-6, epsrel=1e-4, limit=50)[0]

    return (re - 1j*im) * (r1 * r2 * mu_0 / (4 * np.pi))


def inductance_matrix_nonqs(n, m, R, L_own, A, delta, k=None, **kwargs) -> np.ndarray:
    """Неквазистатическая матрица с учетом запаздывания."""
    if k is None:
        omega = getattr(config, 'omega', 2 * np.pi * 68.5e6)
        k = omega / C_LIGHT

    fi = 2 * np.pi / m
    N_total = n * m
    L = np.zeros((N_total, N_total), dtype=complex)
    b_all = np.cumsum(np.insert(np.array(delta), 0, A))
    cache = {}

    for i in range(N_total):
        M_i, N_i = i // n, i % n
        R_i, b_i = R[N_i], b_all[N_i]

        for j in range(i, N_total):
            if i == j:
                L[i, j] = L_own
                continue

            M_j, N_j = j // n, j % n
            R_j, b_j = R[N_j], b_all[N_j]

            delta_M = abs(M_i - M_j)
            if delta_M > m // 2:
                delta_M = m - delta_M
            alpha = delta_M * fi

            cache_key = (N_i, N_j, delta_M) if N_i <= N_j else (N_j, N_i, delta_M)

            if cache_key in cache:
                M_val = cache[cache_key]
            else:
                if delta_M == 0:
                    M_val = L_int_par(0.0, 0.0, abs(b_i - b_j), R_i, R_j, k)
                else:
                    M_val = L_int_angle(b_i, b_j, R_i, R_j, alpha, k)
                cache[cache_key] = M_val

            L[i, j] = M_val
            L[j, i] = M_val

    return L
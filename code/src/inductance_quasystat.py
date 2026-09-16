import numpy as np
from scipy import integrate
import src.config as config

mu_0 = 4 * np.pi * 1e-7
C_LIGHT = 299792458.0


# =========================================================================
# 1. Ваши функции расчета взаимной индуктивности (с защитой от arg < 0)
# =========================================================================

def L_int_par(dx: float, dy: float, dz: float, r1: float, r2: float, k: float, width: float = 0) -> complex:
    """Взаимная индуктивность для параллельных колец."""
    def internal(phi1, phi2):
        A = dz**2 + dx**2 + dy**2 + 2*r2*(dx*np.cos(phi2) + dy*np.sin(phi2)) + r1**2 + r2**2
        B = -2*dx*r1 - 2*r1*r2*np.cos(phi2)
        C = -2*dy*r1 - 2*r1*r2*np.sin(phi2)
        arg = A + B*np.cos(phi1) + C*np.sin(phi1)
        if arg <= 0:
            return 0.0, 0.0
        R = np.sqrt(arg)
        spr = np.cos(phi2 - phi1)
        return spr * np.cos(k*R) / R, spr * np.sin(k*R) / R

    def internal_re(phi1, phi2):
        return internal(phi1, phi2)[0]

    def internal_im(phi1, phi2):
        return internal(phi1, phi2)[1]

    def outer_re(phi2):
        return integrate.quad(internal_re, 0, 2*np.pi, args=(phi2,), epsabs=1e-6, epsrel=1e-4, limit=50)[0]

    def outer_im(phi2):
        return integrate.quad(internal_im, 0, 2*np.pi, args=(phi2,), epsabs=1e-6, epsrel=1e-4, limit=50)[0]

    re = integrate.quad(outer_re, 0, 2*np.pi, epsabs=1e-6, epsrel=1e-4, limit=50)[0]
    im = integrate.quad(outer_im, 0, 2*np.pi, epsabs=1e-6, epsrel=1e-4, limit=50)[0]
    L = (re + 1j*im) * r1 * r2 / (4 * np.pi)
    return L * mu_0


def L_int_ort(dx: float, dy: float, dz: float, r1: float, r2: float, k: float, width: float = 0) -> complex:
    """Взаимная индуктивность для взаимно ортогональных колец (90 градусов)."""
    def internal(phi1, phi2):
        A = dz**2 + dx**2 + dy**2 + 2*r2*(dy*np.sin(phi2) - dz*np.cos(phi2)) + r1**2 + r2**2
        B = 2*dx*r1
        C = 2*dy*r1 + 2*r1*r2*np.sin(phi2)
        arg = A + B*np.cos(phi1) + C*np.sin(phi1)
        if arg <= 0:
            return 0.0, 0.0
        R = np.sqrt(arg)
        spr = np.cos(phi1) * np.cos(phi2)
        return spr * np.cos(k*R) / R, spr * np.sin(k*R) / R

    def internal_re(phi1, phi2):
        return internal(phi1, phi2)[0]

    def internal_im(phi1, phi2):
        return internal(phi1, phi2)[1]

    def outer_re(phi2):
        return integrate.quad(internal_re, 0, 2*np.pi, args=(phi2,), epsabs=1e-6, epsrel=1e-4, limit=50)[0]

    def outer_im(phi2):
        return integrate.quad(internal_im, 0, 2*np.pi, args=(phi2,), epsabs=1e-6, epsrel=1e-4, limit=50)[0]

    re = integrate.quad(outer_re, 0, 2*np.pi, epsabs=1e-6, epsrel=1e-4, limit=50)[0]
    im = integrate.quad(outer_im, 0, 2*np.pi, epsabs=1e-6, epsrel=1e-4, limit=50)[0]
    L = -(re + 1j*im) * r1 * r2 / (4 * np.pi)
    return L * mu_0


def L_int_angle(b1: float, b2: float, r1: float, r2: float, alpha: float, k: float) -> complex:
    """
    Обобщение ваших функций для произвольного угла alpha между стопками.
    При alpha = 0 совпадает с L_int_par, при alpha = pi/2 совпадает с L_int_ort.
    """
    if np.isclose(alpha, 0.0, atol=1e-6):
        return L_int_par(dx=0.0, dy=0.0, dz=abs(b1 - b2), r1=r1, r2=r2, k=k)
    elif np.isclose(alpha, np.pi / 2, atol=1e-6):
        return L_int_ort(dx=b1, dy=b2, dz=0.0, r1=r1, r2=r2, k=k)

    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    def internal(phi1, phi2):
        x2 = b2 * cos_a - r2 * sin_a * np.cos(phi2)
        y2 = b2 * sin_a + r2 * cos_a * np.cos(phi2)
        z2 = r2 * np.sin(phi2)

        A = (x2 - b1)**2 + y2**2 + z2**2 + r1**2
        B = -2 * r1 * y2
        C = -2 * r1 * z2
        arg = A + B * np.cos(phi1) + C * np.sin(phi1)
        if arg <= 0:
            return 0.0, 0.0
        R = np.sqrt(arg)
        spr = cos_a * np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2)
        return spr * np.cos(k*R) / R, spr * np.sin(k*R) / R

    def internal_re(phi1, phi2):
        return internal(phi1, phi2)[0]

    def internal_im(phi1, phi2):
        return internal(phi1, phi2)[1]

    def outer_re(phi2):
        return integrate.quad(internal_re, 0, 2*np.pi, args=(phi2,), epsabs=1e-6, epsrel=1e-4, limit=50)[0]

    def outer_im(phi2):
        return integrate.quad(internal_im, 0, 2*np.pi, args=(phi2,), epsabs=1e-6, epsrel=1e-4, limit=50)[0]

    re = integrate.quad(outer_re, 0, 2*np.pi, epsabs=1e-6, epsrel=1e-4, limit=50)[0]
    im = integrate.quad(outer_im, 0, 2*np.pi, epsabs=1e-6, epsrel=1e-4, limit=50)[0]
    L = (re + 1j*im) * r1 * r2 / (4 * np.pi)
    return L * mu_0


# =========================================================================
# 2. Сборка матрицы индуктивностей L
# =========================================================================

def inductance_matrix(n, m, R, L_own, A, delta, k=None, **kwargs):
    """
    Расчет матрицы индуктивностей n*m x n*m.
    Принимает **kwargs (all_points, normals, N_seg), чтобы не ломать вызовы в main.py и optimization.py.
    """
    if k is None:
        omega = getattr(config, 'omega', 2 * np.pi * 68.5e6)
        k = omega / C_LIGHT

    fi = 2 * np.pi / m
    N_total = n * m
    L = np.zeros((N_total, N_total), dtype=complex)

    # Расстояния центров колец от центра системы вдоль луча стопки
    x_shifts = np.insert(np.array(delta), 0, A)
    b_all = np.cumsum(x_shifts)

    # Кэш для ускорения: индуктивность между одинаковыми парами колец на одинаковых углах
    cache = {}

    for i in range(N_total):
        M_i = i // n  # Индекс стопки
        N_i = i % n   # Индекс кольца в стопке
        R_i = R[N_i]
        b_i = b_all[N_i]

        for j in range(i, N_total):
            if i == j:
                L[i, j] = L_own
                continue

            M_j = j // n
            N_j = j % n
            R_j = R[N_j]
            b_j = b_all[N_j]

            # Угол между стопками
            delta_M = abs(M_i - M_j)
            if delta_M > m // 2:
                delta_M = m - delta_M
            alpha = delta_M * fi

            cache_key = (N_i, N_j, delta_M) if N_i <= N_j else (N_j, N_i, delta_M)

            if cache_key in cache:
                M_val = cache[cache_key]
            else:
                if delta_M == 0:
                    # Кольца в одной стопке (коаксиальные, параллельные)
                    M_val = L_int_par(dx=0.0, dy=0.0, dz=abs(b_i - b_j), r1=R_i, r2=R_j, k=k)
                else:
                    # Кольца в разных стопках под углом alpha
                    M_val = L_int_angle(b1=b_i, b2=b_j, r1=R_i, r2=R_j, alpha=alpha, k=k)

                cache[cache_key] = M_val

            L[i, j] = M_val
            L[j, i] = M_val

    return L
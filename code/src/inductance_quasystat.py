# src/inductance_quasystat.py
import numpy as np
import scipy.integrate
import math
from scipy import special

mu_0 = 4 * np.pi * 1e-7


def inductance_coaxial(R1: float, R2: float, dz: float) -> float:
    """Точная аналитическая формула Максвелла для соосных колец (без интегрирования)."""
    if dz < 1e-12 and math.isclose(R1, R2, rel_tol=1e-5):
        return 0.0
    k2 = (4 * R1 * R2) / ((R1 + R2)**2 + dz**2)
    k2 = np.clip(k2, 0.0, 1.0 - 1e-12)
    k = math.sqrt(k2)
    K = special.ellipk(k2)
    E = special.ellipe(k2)
    return mu_0 * math.sqrt(R1 * R2) * ((2.0 / k - k) * K - (2.0 / k) * E)


def inductance_inclined(R1: float, R2: float, b1: float, b2: float, alpha: float) -> float:
    """Расчет M через циркуляцию векторного потенциала A (1D-интеграл по второму кольцу)."""
    def integrand(theta):
        z2 = b1 - b2 * math.cos(alpha)
        x2 = b2 * math.sin(alpha)
        x = -x2 - R2 * math.cos(alpha) * math.sin(theta)
        y = R2 * math.cos(theta)
        z = z2 + R2 * math.sin(alpha) * math.cos(theta)

        rho = math.sqrt(x * x + y * y)
        if rho < 1e-12:
            return 0.0

        k2 = (4 * R1 * rho) / ((R1 + rho)**2 + z**2)
        k2 = np.clip(k2, 0.0, 1.0 - 1e-12)
        k = math.sqrt(k2)

        numerator = R2 * math.cos(alpha) + x2 * math.sin(theta)
        denominator = math.sqrt((x2 + R2 * math.cos(alpha) * math.sin(theta))**2 + (R2 * math.cos(theta))**2)
        if denominator < 1e-12:
            return 0.0

        coeff = numerator / denominator
        factor = (mu_0 / np.pi) * (1.0 / (k * math.sqrt(rho))) * (
            (1.0 - k2 / 2.0) * special.ellipk(k2) - special.ellipe(k2)
        ) * coeff
        return factor

    res, _ = scipy.integrate.quad(integrand, 0, 2 * np.pi, epsabs=1e-8, epsrel=1e-6, limit=50)
    return R2 * math.sqrt(R1) * res


def inductance_matrix_quasystat(n, m, R, L_own, A, delta, **kwargs) -> np.ndarray:
    """Быстрая матрица индуктивностей в квазистатическом приближении."""
    fi = 2 * np.pi / m
    N_total = n * m
    L = np.zeros((N_total, N_total), dtype=complex)

    x_shifts = np.insert(np.array(delta), 0, A)
    b_all = np.cumsum(x_shifts)
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
                    M_val = inductance_coaxial(R_i, R_j, abs(b_i - b_j))
                else:
                    M_val = inductance_inclined(R_i, R_j, b_i, b_j, alpha)
                cache[cache_key] = M_val

            L[i, j] = M_val
            L[j, i] = M_val

    return L
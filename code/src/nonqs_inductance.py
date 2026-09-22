# src/nonqs_inductance.py
import numpy as np
from scipy.integrate import dblquad
from src.geometry import points_on_rings_one_stack, stack_basis

mu0 = 4 * np.pi * 1e-7
c = 299792458.0


# ---------------------------------------------------------------------------
# 1. Геометрия: точки + касательные (расширение points_on_rings_general)
# ---------------------------------------------------------------------------
def points_and_tangents_on_rings_general(delta, n, A, N, R, m):
    """
    Возвращает:
        coords   : (m*n*N, 3)
        tangents : (m*n*N, 3)
        normals  : (m, 3)
    """
    fi = 2 * np.pi / m
    base_part = points_on_rings_one_stack(delta, n, A, N, R)
    base_normal = np.array([-1.0, 0.0, 0.0])

    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)

    delta_arr = np.asarray(delta)
    x_shifts = np.insert(delta_arr, 0, A)
    x_centers = np.cumsum(x_shifts)
    R_arr = np.asarray(R)

    base_tangents = []
    for i in range(n):
        r = R_arr[i]
        tx = np.zeros_like(theta)
        ty = -r * np.sin(theta)
        tz =  r * np.cos(theta)
        base_tangents.append(np.vstack([tx, ty, tz]).T)
    base_tangents = np.vstack(base_tangents)

    angles = np.arange(m) * fi
    coords_list, tangents_list, normals_list = [], [], []
    for angle in angles:
        ca, sa = np.cos(angle), np.sin(angle)
        Rz = np.array([
            [ca, -sa, 0.0],
            [sa,  ca, 0.0],
            [0.0, 0.0, 1.0],
        ])
        coords_list.append(base_part @ Rz.T)
        tangents_list.append(base_tangents @ Rz.T)
        normals_list.append(base_normal @ Rz.T)

    return (np.vstack(coords_list),
            np.vstack(tangents_list),
            np.array(normals_list))


"""
def fit_ring_geometry(coords_block):
    
    coords_block : (N, 3), точки одного кольца, равномерно по theta.
    Возвращает center (3,), e1 (3,), e2 (3,), radius (float).
    
    center = coords_block.mean(axis=0)
    v0 = coords_block[0] - center
    r = np.linalg.norm(v0)
    e1 = v0 / r
    v1 = coords_block[1] - center
    perp = v1 - (v1 @ e1) * e1
    e2 = perp / np.linalg.norm(perp)
    return center, e1, e2, r
"""

# ---------------------------------------------------------------------------
# 3. Не квазистатическая M между двумя кольцами (по точкам)
# ---------------------------------------------------------------------------
def mutual_inductance_rings_nonqs(
    c1, e1a, e1b, r1,   # аналитические параметры первого кольца
    c2, e2a, e2b, r2,   # аналитические параметры второго кольца
    f,
    epsabs=1e-10, epsrel=1e-8,
):
    """
    Не квазистатическая взаимная индуктивность между двумя кольцами.
    
    Параметры:
        c1, c2       : (3,) центры колец
        e1a, e1b     : (3,) ортонормированный базис плоскости первого кольца
        e2a, e2b     : (3,) ортонормированный базис плоскости второго кольца
        r1, r2       : радиусы колец
        f            : частота (Гц)
    """
    k = 2.0 * np.pi * f / c

    def base_kernel(theta1, theta2):
        """Возвращает (dl1·dl2, 1/R, cos(kR), sin(kR)) — всё действительное."""
        ct1, st1 = np.cos(theta1), np.sin(theta1)
        ct2, st2 = np.cos(theta2), np.sin(theta2)
        p1 = c1 + r1 * (ct1 * e1a + st1 * e1b)
        p2 = c2 + r2 * (ct2 * e2a + st2 * e2b)
        t1 = r1 * (-st1 * e1a + ct1 * e1b)
        t2 = r2 * (-st2 * e2a + ct2 * e2b)
        d = p2 - p1
        R_dist = np.sqrt(d @ d)
        if R_dist < 1e-15:
            return 0.0, 0.0, 1.0, 0.0
        dl_dot = t1 @ t2
        cos_kR = np.cos(k * R_dist)
        sin_kR = np.sin(k * R_dist)
        return dl_dot, 1.0 / R_dist, cos_kR, sin_kR

    # dblquad требует func(y, x) — первая переменная внутренняя.
    def integrand_real(theta2, theta1):
        dl_dot, inv_R, cos_kR, _ = base_kernel(theta1, theta2)
        return dl_dot * inv_R * cos_kR

    def integrand_imag(theta2, theta1):
        dl_dot, inv_R, _, sin_kR = base_kernel(theta1, theta2)
        return -dl_dot * inv_R * sin_kR   # exp(-jkR) = cos(kR) - j sin(kR)

    re_int, _ = dblquad(
        integrand_real, 0.0, 2*np.pi,
        lambda t1: 0.0, lambda t1: 2*np.pi,
        epsabs=epsabs, epsrel=epsrel,
    )
    im_int, _ = dblquad(
        integrand_imag, 0.0, 2*np.pi,
        lambda t1: 0.0, lambda t1: 2*np.pi,
        epsabs=epsabs, epsrel=epsrel,
    )
    return mu0 / (4 * np.pi) * (re_int + 1j * im_int)


# ---------------------------------------------------------------------------
# 4. Полная матрица (аналог inductance_matrix, но неquasi-static)
# ---------------------------------------------------------------------------
def inductance_matrix_nonqs(
    ring_centers, normals, e1, e2, radii,
    n, m, L_own, f=0.0,
    epsabs=1e-10, epsrel=1e-8,
):
    mn = m * n
    L = np.zeros((mn, mn), dtype=complex)
    for i in range(mn):
        i_stack, i_ring = i // n, i % n
        for j in range(i, mn):
            j_stack, j_ring = j // n, j % n
            if i == j:
                L[i, j] = L_own
                continue
            M = mutual_inductance_rings_nonqs(
                c1=ring_centers[i], e1a=e1[i_stack], e1b=e2[i_stack], r1=radii[i_ring],
                c2=ring_centers[j], e2a=e1[j_stack], e2b=e2[j_stack], r2=radii[j_ring],
                f=f, epsabs=epsabs, epsrel=epsrel,
            )
            sign = np.sign(np.dot(normals[i_stack], normals[j_stack]))
            L[i, j] = sign * M
            L[j, i] = L[i, j]
    return L
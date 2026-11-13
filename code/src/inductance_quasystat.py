import numpy as np
from scipy import integrate, special

mu_0 = 4 * np.pi * 1e-7


def mutual_inductance_inclined_fast(R1, R2, c1, e1a, e1b, c2, e2a, e2b):
    n1 = np.cross(e1a, e1b)
    n1 = n1 / np.linalg.norm(n1)
    n2 = np.cross(e2a, e2b)
    n2 = n2 / np.linalg.norm(n2)

    d = c2 - c1
    x2_loc = np.dot(d, e1a)
    y2_loc = np.dot(d, e1b)
    z2_loc = np.dot(d, n1)

    n2x = np.dot(n2, e1a)
    n2y = np.dot(n2, e1b)
    n2z = np.dot(n2, n1)

    if abs(y2_loc) > 1e-10 or abs(n2y) > 1e-10:
        phi = np.arctan2(y2_loc, x2_loc)
        ca, sa = np.cos(phi), np.sin(phi)
        e1a_new = ca * e1a + sa * e1b
        e1b_new = -sa * e1a + ca * e1b
        e1a, e1b = e1a_new, e1b_new
        x2_loc = np.dot(d, e1a)
        y2_loc = np.dot(d, e1b)
        n2x = np.dot(n2, e1a)
        n2y = np.dot(n2, e1b)

    x2_doc = -x2_loc
    z2_doc = z2_loc
    alpha_doc = np.arccos(np.clip(n2z, -1.0, 1.0))
    if n2x < 0:
        alpha_doc = -alpha_doc

    def integrand(theta):
        x = -x2_doc - R2 * np.cos(alpha_doc) * np.sin(theta)
        y = R2 * np.cos(theta)
        z = z2_doc + R2 * np.sin(alpha_doc) * np.sin(theta)

        rho = np.sqrt(x**2 + y**2)
        if rho < 1e-15:
            return 0.0

        k2 = 4 * R1 * rho / ((R1 + rho)**2 + z**2)
        k2 = min(max(k2, 0.0), 1.0 - 1e-12)
        k = np.sqrt(k2)

        K = special.ellipk(k2)
        E = special.ellipe(k2)

        numerator = R2 * np.cos(alpha_doc) + x2_doc * np.sin(theta)
        denominator = np.sqrt((x2_doc + R2 * np.cos(alpha_doc) * np.sin(theta))**2
                              + (R2 * np.cos(theta))**2)
        if denominator < 1e-15:
            return 0.0
        coeff = numerator / denominator

        factor = (mu_0 / np.pi) * (1.0 / (k * np.sqrt(rho))) * (
            (1.0 - k2 / 2.0) * K - E
        )
        return factor * coeff

    res, _ = integrate.quad(integrand, 0, 2 * np.pi,
                            epsabs=1e-6, epsrel=1e-4, limit=100)
    return R2 * np.sqrt(R1) * res


def self_inductance_ring(R, wire_radius):
    a = max(wire_radius, 1e-6)
    return mu_0 * R * (np.log(8.0 * R / a) - 2.0)


def mutual_inductance_rings_nonqs(c1, e1a, e1b, r1,
                                  c2, e2a, e2b, r2,
                                  f=0.0, epsabs=1e-8, epsrel=1e-6):
    M = mutual_inductance_inclined_fast(r1, r2, c1, e1a, e1b, c2, e2a, e2b)
    return complex(M, 0.0)


_GEOM = {}


def set_geometry(m, n, ring_centers, e1, e2, radii, L_own):
    _GEOM['m'] = int(m)
    _GEOM['n'] = int(n)
    _GEOM['ring_centers'] = np.asarray(ring_centers, dtype=float)
    _GEOM['e1'] = np.asarray(e1, dtype=float)
    _GEOM['e2'] = np.asarray(e2, dtype=float)
    _GEOM['radii'] = np.asarray(radii, dtype=float)
    _GEOM['L_own'] = L_own


def inductance_matrix_quasystat(f_hz):
    m = _GEOM['m']
    n = _GEOM['n']
    ring_centers = _GEOM['ring_centers']
    e1 = _GEOM['e1']
    e2 = _GEOM['e2']
    radii = _GEOM['radii']
    L_own = _GEOM['L_own']

    mn = m * n
    L = np.zeros((mn, mn), dtype=complex)

    for i in range(mn):
        i_s, i_r = i // n, i % n
        for j in range(i, mn):
            j_s, j_r = j // n, j % n
            if i == j:
                L[i, j] = L_own
                continue
            M = mutual_inductance_rings_nonqs(
                c1=ring_centers[i], e1a=e1[i_s], e1b=e2[i_s], r1=radii[i_r],
                c2=ring_centers[j], e2a=e1[j_s], e2b=e2[j_s], r2=radii[j_r],
                f=f_hz, epsabs=1e-8, epsrel=1e-6,
            )
            L[i, j] = M
            L[j, i] = M
    return L
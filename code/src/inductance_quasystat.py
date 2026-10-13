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

    res, _ = integrate.quad(integrand, 0, 2*np.pi,
                             epsabs=1e-10, epsrel=1e-8, limit=200)
    return R2 * np.sqrt(R1) * res
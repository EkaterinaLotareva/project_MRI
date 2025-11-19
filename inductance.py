import numpy as np
import scipy
from math import *
from scipy import special

mu = 1

def function(theta, R1, R2, b1, b2, alpha):

    z2 = b1 - b2 * cos(alpha)
    x2 = b2 * sin(alpha)
    x = -x2 - R2 * cos(alpha) * sin(theta)
    y = R2 * cos(theta)
    z = z2 + R2 * sin(alpha) * cos(theta)
    rho = sqrt(x ** 2 + y ** 2)
    k = sqrt((4*R1*rho)/((R1+rho)**2 + z**2))

    return ((2*mu*sqrt(R1))/(k*sqrt(rho)*4*pi)) * ((1-k**2/2)*special.ellipk(k) - special.ellipe(k)) * \
           ((R2*cos(alpha) + x2*sin(theta))/(sqrt((x2+R2*cos(alpha)*sin(theta))**2 + (R2*cos(theta))**2))) * R2



def inductance(R1, R2, b1, b2, alpha):
    return scipy.integrate.quad(function, 0, 2*pi, args = (R1, R2, b1, b2, alpha))

def inductance_matrix(n, m, R, A, a):

    fi = pi/(m-1)
    L = np.zeros((n*m, n*m))

    for i in range(0, n*m):
        for j in range(0, n*m):
            M_i = i // n    # номер стопки
            N_i = i % n     # номер кольца в стопке
            M_j = j // n
            N_j = j % n
            L[i,j] = inductance(R[N_i], R[N_j], A - a*N_i, A - a*N_j, fi * abs(M_i - M_j))
    return L




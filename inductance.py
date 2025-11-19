import numpy
import scipy
from math import *
from scipy import special

def function(theta, R1, R2, b1, b2, alpha, mu):

    z2 = b1 - b2 * cos(alpha)
    x2 = b2 * sin(alpha)
    x = -x2 - R2 * cos(alpha) * sin(theta)
    y = R2 * cos(theta)
    z = z2 + R2 * sin(alpha) * cos(theta)
    rho = sqrt(x ** 2 + y ** 2)
    k = sqrt((4*R1*rho)/((R1+rho)**2 + z**2))

    return ((2*mu*sqrt(R1))/(k*sqrt(rho)*4*pi)) * ((1-k**2/2)*special.ellipk(k) - special.ellipe(k)) * \
           ((R2*cos(alpha) + x2*sin(theta))/(sqrt((x2+R2*cos(alpha)*sin(theta))**2 + (R2*cos(theta))**2))) * R2



def inductance(R1, R2, b1, b2, alpha, mu):
    scipy.integrate.quad(function, 0, 2*pi, args = (R1, R2, b1, b2, alpha, mu))




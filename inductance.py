import numpy as np
import scipy
from math import *
from scipy import special



def inductance(R1, R2, b1, b2, alpha):
    if b1 == b2 and alpha == 0:
        print('совпадение колец')
    if math.sqrt(b1**2 + b2**2 - b1*b2*math.cos(alpha)) <= R1+R2:
        print('пересечение колец')
    mu=4 * np.pi * 1e-7
    def integrand(theta, R1, R2, b1, b2, alpha, mu, eps=1e-12):
        # геометрия точки на контуре R2
        z2 = b1 - b2 * math.cos(alpha)
        x2 = b2 * math.sin(alpha)
        x = -x2 - R2 * math.cos(alpha) * math.sin(theta)
        y = R2 * math.cos(theta)
        z = z2 + R2 * math.sin(alpha) * math.cos(theta)

        rho = math.sqrt(x * x + y * y)

        # Защита от нулевого rho (точка на оси) и от очень маленького k
        if rho < eps:
            rho = eps

        k = math.sqrt((4 * R1 * rho) / ((R1 + rho) ** 2 + z ** 2))

        if math.isclose(alpha, 0.0, abs_tol=eps) and math.isclose(b1, b2, abs_tol=eps):
            return 0.0

        if k < eps:
            return 0.0

        numerator = (R2 * math.cos(alpha) + x2 * math.sin(theta))
        denominator = math.sqrt((x2 + R2 * math.cos(alpha) * math.sin(theta)) ** 2 + (R2 * math.cos(theta)) ** 2)
        if denominator < eps:
            coefficient = 0.0
        else:
            coefficient = numerator / denominator

        factor = (mu / (2 * np.pi)) * (1 / (k * math.sqrt(rho))) * ((1.0 - k * k / 2.0) * special.ellipk(k * k) - special.ellipe(k * k)) * coefficient

        return factor * coefficient

        # Вычисление интеграла
    result = scipy.integrate.quad(
                                 integrand,
                                  0,
                                        2 * np.pi,
                                        args=(R1, R2, b1, b2, alpha, mu),
                                        epsabs=1e-12,
                                        epsrel=1e-12,
                                        limit=100 )

    return  R2 * math.sqrt(R1) * result[0]


#Получение матрицы взаимных индуктивностей
def inductance_matrix(n, m, R, r, A, delta):

    fi = 2*np.pi/m
    L = np.zeros((n*m, n*m))
    r = r.reshape(n*m, 1)

    for i in range(n * m):
        M_i = i // n  # номер стопки
        N_i = i % n  # номер кольца в стопке
        R_i = R[N_i]
        r_i = r[N_i]
        b_i = A + sum(delta[0:N_i])

        for j in range(i, n * m):  # только j >= i
            M_j = j // n
            N_j = j % n
            R_j = R[N_j]
            r_j = r[N_j]
            b_j = A + sum(delta[0:N_j])

            alpha = fi * abs(M_i - M_j)
            if i == j:
              L[i, j] = 4*np.pi*1e-7*float(R_j)*(np.log(8*float(R_j)/float(r_j)) - 2)
            else:
              L[i, j] = inductance(R_i, R_j, b_i, b_j, alpha)
              L[j, i] = L[i, j]

    return L

#Получение матрицы импедансов без учета взаимных индуктивностей

def demo_Z_self_matrix(sigma, r, C, n, m, R):
    demo_Z_self = np.zeros((n*m, n*m))
    r = r.reshape(n*m, 1)
    C = C.reshape(n*m, 1)
    for i in range(n * m):
        for j in range(i, n * m):  # только j >= i
            M_j = j // n
            N_j = j % n
            C_j = C[N_j]
            r_j = r[N_j]
            R_j = R[N_j]
            if i == j:
              demo_Z_self[i, j] = 1/sigma * 1/(np.pi*r_j**2*1e6) * 2*np.pi * R_j
            else:
              demo_Z_self[j, i] = 0
              demo_Z_self[j, i] = demo_Z_self[i, j]

    return demo_Z_self

#Создание вектора напряжений (запитывается каждое n-ое кольцо)

def create_U(n, m, U_A=1.0, f=63.8e6):

    N = n * m
    U = np.zeros(N, dtype=complex)
    indices_to_feed = np.arange(0, N, n)
    U[indices_to_feed] = U_A

    return U

#Функция получения вектора токов через решение матричного ур-я

def calc_I(Z_self, U, omega, L): # Z - матрица импедансов, U - матрица напряжений


    if Z_self.dtype != np.complex128:
        Z_self = Z_self.astype(np.complex128, copy=False)
    if U.dtype != np.complex128:
        U = U.astype(np.complex128, copy=False)
    if L.dtype != np.float64:
        L = L.astype(np.float64, copy=False)

    Z_self_matrix = Z_self.T + np.eye(n*m)
    Z = Z_self_matrix - 1j*omega*L
    I = np.linalg.solve(Z, U)

    return I




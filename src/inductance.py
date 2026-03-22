import numpy as np
import scipy
import math
from scipy import special
<<<<<<< HEAD
from src.geometry import points_on_rings_general, ring_center_general
=======
from src.geometry import points_on_rings_general, ring_center_general, get_ring_normal
>>>>>>> 1de6c2c92a56bf2303adee686b508e19c28d6d87



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

        return factor 

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
<<<<<<< HEAD
def inductance_matrix(n, m, R, L_own, A, delta, all_points, normals, N_seg):

    # R - массив длины n (параметры одной стопки)
   
    fi = 2 * np.pi / m
    L = np.zeros((n * m, n * m))

    for i in range(n * m):
=======
def inductance_matrix(n, m, R, L_own, A, delta, all_points, N_seg):

    # R - массив длины n (параметры одной стопки)
   
    fi = 2 * np.pi / m
    N_total = n * m
    L = np.zeros((N_total, N_total))

    for i in range(N_total):
        pts_i = all_points[i*N_seg : (i+1)*N_seg]
        n_i = get_ring_normal(pts_i) 

>>>>>>> 1de6c2c92a56bf2303adee686b508e19c28d6d87
        M_i = i // n  # номер стопки
        N_i = i % n   # номер кольца в стопке

        R_i = R[N_i]
        b_i = A + np.sum(delta[0:N_i])
        n_i = normals[M_i]

        for j in range(i, n * m):
        
            M_j = j // n
            N_j = j % n

            R_j = R[N_j]
            b_j = A + np.sum(delta[0:N_j])
<<<<<<< HEAD
            n_j = normals[M_j]
=======

            pts_j = all_points[j * N_seg : (j + 1) * N_seg]
            n_j = get_ring_normal(pts_j)
>>>>>>> 1de6c2c92a56bf2303adee686b508e19c28d6d87

            if i == j:
                L[i, j] = L_own
            else:
<<<<<<< HEAD
                cos_alpha = np.dot(n_i, n_j)
                alpha = np.arccos(np.clip(cos_alpha, -1.0, -1.0))
    
                sign = np.sign(cos_alpha)
                M = inductance(R_i, R_j, b_i, b_j, alpha)
                L[i, j] = sign * M
                L[j, i] = L[i, j]
=======
                # Взаимная индуктивность
                cos_alpha = np.dot(n_i, n_j)
                alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
                L[i, j] = inductance(R_i, R_j, b_i, b_j, alpha)
                L[j, i] = L[i, j]  # Симметрия
>>>>>>> 1de6c2c92a56bf2303adee686b508e19c28d6d87

    return L


from unittest import result

import numpy as np
import scipy
import math
from scipy import special
from src.geometry import points_on_rings_general, ring_center_general
from src.shoora import L_parallel

mu=4 * np.pi * 1e-7

def inductance(R1, R2, b1, b2, alpha):
    print(round(np.abs(b2-b1), 4), round(R1, 4))
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

        factor = (mu / (np.pi)) * (1 / (k * math.sqrt(rho))) * ((1.0 - k * k / 2.0) * special.ellipk(k * k) - special.ellipe(k * k)) * coefficient
        
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
    if True:
        theta = 0  # Выбираем произвольное значение для theta, например, 0
        eps = 1e-12
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

        factor = (mu / (np.pi)) * (1 / (k * math.sqrt(rho))) * ((1.0 - k * k / 2.0) * special.ellipk(k * k) - special.ellipe(k * k)) * coefficient
        """
        print("coefficient:", coefficient)
        print("factor without coefficient:", factor / coefficient)
        print("nominator:", numerator)
        print("denominator:", denominator)
        print("Rho term", 1 / (k * math.sqrt(rho)))
        print("ellipk K", (1.0 - k * k / 2.0) * special.ellipk(k * k))
        print("ellipk E", special.ellipe(k * k))
    print(f"Интеграл для R1={R1}, (b1 - b2)={round(np.abs(b1-b2), 4)}, alpha={alpha}: {result[0]} (ошибка: {result[1]})")
    """
    return  R2 * math.sqrt(R1) * result[0]


#Получение матрицы взаимных индуктивностей
def inductance_matrix(n, m, R, L_own, A, delta, all_points, normals, N_seg):

    # R - массив длины n (параметры одной стопки)
   
    fi = 2 * np.pi / m
    L = np.zeros((n * m, n * m))

    for i in range(n * m):
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
            n_j = normals[M_j]

            if i == j:
                L[i, j] = L_own
            else:
                if M_i == M_j:
                    # Кольца в одной стопке, коаксиальные
                    cos_alpha = np.dot(n_i, n_j)
                    alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))
                    sign = np.sign(cos_alpha)
                    M = inductance(R_i, R_j, b_i, b_j, alpha)
                else:
                    # Кольца в разных стопках, параллельные
                    center_i = ring_center_general(delta, A, n, fi, M_i, N_i)
                    center_j = ring_center_general(delta, A, n, fi, M_j, N_j)
                    dx = center_i[0] - center_j[0]
                    dy = center_i[1] - center_j[1]
                    dz = center_i[2] - center_j[2]
                    sign = np.sign(np.dot(n_i, n_j))
                    M = L_parallel(dx, dy, dz, R_i, R_j)
                # print(f"Взаимная индуктивность между кольцом {N_i} стопки {M_i} и кольцом {N_j} стопки {M_j}: {M}") 
                L[i, j] = sign * M
                L[j, i] = L[i, j]

    return L


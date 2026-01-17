import numpy as np
import math

#Получение матрицы импедансов без учета взаимных индуктивностей

def Z_self_matrix(rho, r, C, n, m, R):
    Z_self = np.zeros((n*m, n*m))
    C = np.array(C)
    C = C.reshape(n*m, 1)
    for i in range(n * m):
        for j in range(i, n * m):  # только j >= i
            M_j = j // n
            N_j = j % n
            C_j = C[N_j]
            R_j = R[N_j]
            if i == j:
              Z_self[i, j] = rho * 1/(np.pi*r**2*1e6) * 2*np.pi * R_j
            else:
              Z_self[j, i] = 0
              Z_self[j, i] = Z_self[i, j]

    return Z_self

def generate_voltage_array(U_0, m, n, phi_0=0):
    """
    Генерирует массив напряжений, где напряжение есть только в первом элементе
    каждой стопки (группы) с круговым фазовым сдвигом 2*pi/M между стопками.
    Остальные элементы имеют нулевое напряжение.

    Аргументы:
    U_0 (float): Амплитуда напряжения в элементах с напряжением.
    m (int): Количество стопок (групп).
    n (int): Количество элементов в каждой стопке.
    phi_0 (float): Начальная фаза (по умолчанию 0).

    Возвращает:
    numpy.ndarray: Одномерный массив комплексных чисел, представляющих напряжения.
    """
    V_array = np.zeros(m * n, dtype=complex)

    delta_phi = (2 * math.pi) / m

    for i in range(m):
        idx = i * n
        phi = phi_0 - i * delta_phi
        V_array[idx] = U_0 * np.exp(1j * phi)

    return V_array

#Функция получения вектора токов через решение матричного ур-я

def calc_I(Z_self, U, omega, L, n, m): # Z - матрица импедансов, U - матрица напряжений


    if Z_self.dtype != np.complex128:
        Z_self = Z_self.astype(np.complex128, copy=False)
    if U.dtype != np.complex128:
        U = U.astype(np.complex128, copy=False)
    if L.dtype != np.float64:
        L = L.astype(np.float64, copy=False)

    Z_self_matrix = Z_self.T + np.eye(n*m)
    Z = Z_self_matrix - 1j*omega*L
    I = np.linalg.solve(Z, U)
    I = I.reshape(m, n)

    return I




import numpy as np
import math

#Получение матрицы импедансов без учета взаимных индуктивностей

def Z_self_matrix(r, C, n, m, R, omega):
    """
    r - сопротивление кольца
    C - ёмкость конденсаторов (массив длины n)
    R - радиусы колец (массив длины n)
    omega - частота
    """
    N = n * m
    Z_self = np.zeros((N, N), dtype=complex)

    for i in range(N):
        ring_idx = i % n  # Номер типа кольца (0..n-1)
        
        # Ёмкостное сопротивление: Z_C = i / (omega * C)
        if C[ring_idx] > 0:
            Z_C = 1j / (omega * C[ring_idx])
        else:
            Z_C = 0

        # Диагональ: R + Z_C
        Z_self[i, i] = r + Z_C

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
    
    """
    Z_self - матрица с R и C (из исправленной функции выше)
    U - вектор напряжений
    omega - циклическая частота
    L - полная матрица индуктивностей (включая взаимные L_ij)
    n - количество колец в стопке
    m - количество стопок
    """

    Z_self = Z_self.astype(np.complex128, copy=False)
    U = U.astype(np.complex128, copy=False)
    L = L.astype(np.complex128, copy=False)
    
    invalid_L = ~np.isfinite(L)
    if np.any(invalid_L):
        L[invalid_L] = 1e-12  

    invalid_Z = ~np.isfinite(Z_self)
    if np.any(invalid_Z):
        Z_self[invalid_Z] = 1e-6  
    try:
        Z = Z_self - 1j * omega * L
        
        Z_abs = np.abs(Z)
        if np.any(Z_abs < 1e-12) or np.any(~np.isfinite(Z_abs)):
            Z = Z + 1e-8 * np.eye(len(Z))
        
    except Exception as e:
        print(f"Ошибка при расчёте Z: {e}")
        raise

    try:
        I_flat = np.linalg.solve(Z, U)  
    except np.linalg.LinAlgError as e:
        print(f"Ошибка при решении системы: {e}")
        print(f"det(Z) = {np.linalg.det(Z)}")
        raise

    
    I_matrix = I_flat.reshape(m, n)
    
    
    if not np.all(np.isfinite(I_matrix)):
        print("Предупреждение: токи содержат NaN/Inf значения")
        I_matrix = np.nan_to_num(I_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return I_matrix

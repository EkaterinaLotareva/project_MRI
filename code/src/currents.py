import numpy as np
import math

def Z_self_matrix(r, C, n, m, R, omega):
    N = n * m
    Z_self = np.zeros((N, N), dtype=complex)
    C_arr = np.asarray(C)

    for i in range(N):
        c_val = C_arr[i] if C_arr.size == N else C_arr[i % n]
        # Z_C = 1 / (j * omega * C) = -1j / (omega * C)
        Z_C = -1j / (omega * c_val) if c_val > 0 else 0.0
        Z_self[i, i] = r + Z_C

    return Z_self

def generate_voltage_array(U_0, m, n, phi_0=0):
    V_array = np.zeros(m * n, dtype=complex)
    delta_phi = (2 * math.pi) / m
    for i in range(m):
        idx = i * n
        phi = phi_0 - i * delta_phi
        V_array[idx] = U_0 * np.exp(1j * phi)
    return V_array

def calc_I(Z_self, U, omega, L, n=None, m=None):
    """
    Решает уравнение Z * I = U, где Z = Z_self + 1j * omega * L.
    Возвращает матрицу токов формы (m, n).
    """
    Z = Z_self + 1j * omega * L
    
    # Регуляризация для сингулярных случаев
    if np.any(np.abs(np.diag(Z)) < 1e-12):
        Z = Z + 1e-8 * np.eye(len(Z))

    I_flat = np.linalg.solve(Z, U)

    if n is not None and m is not None:
        return I_flat.reshape((m, n))
    return I_flat
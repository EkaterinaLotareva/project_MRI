import numpy as np

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




import numpy as np

def impedance(R, C, L, w):

    if len(R) == L.shape[0]:
        R = np.array(R)
    else:
        raise ValueError(f"Длина R ({len(R)}) должна совпадать с размерами матрицы L")

    if len(C) == L.shape[0]:
        C = np.array(C)
    else:
        raise ValueError(f"Длина C ({len(C)}) должна совпадать с размерами матрицы L")

    impedance = -1j*w*L + np.diag(R) + np.diag((1/C))*(1j/w)
    return impedance

def currents(impedance, U):

    if len(U) == impedance.shape[0]:
        U = np.array(U).T
    else:
        raise ValueError(f"Длина R ({len(U)}) должна совпадать с размерами матрицы импедансов")

    I = np.linalg.solve(impedance, U)
    I = I.T
    return I
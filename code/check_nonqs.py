import numpy as np
from scipy.special import ellipk, ellipe
from src.nonqs_inductance import inductance_matrix_nonqs

mu0 = 4 * np.pi * 1e-7


# --- Аналитическая (квазистатическая) M двух соосных колец ---
def M_analytic_coaxial(a, b, d):
    """
    a, b : радиусы
    d    : расстояние между плоскостями колец (вдоль общей оси)
    """
    alpha2 = d * d + (a + b) ** 2
    k2 = 4 * a * b / alpha2
    K, E = ellipk(k2), ellipe(k2)
    return mu0 * np.sqrt(a * b) * (
        (2.0 / np.sqrt(k2) - np.sqrt(k2)) * K
        - (2.0 / np.sqrt(k2)) * E
    )


# ===========================================================================
# ТЕСТ 1: два соосных кольца в одной стопке (m=1, n=2)
# ===========================================================================
print("=== Тест 1: два соосных кольца (m=1, n=2) ===")

R1 = 0.10         # радиус первого кольца
R2 = 0.08         # радиус второго кольца
gap = 0.05        # расстояние между кольцами вдоль X
A = 0.0           # начальное смещение (первое кольцо в x=0)
delta = np.array([gap])   # смещение от кольца 0 к кольцу 1
n = 2
m = 1
R = np.array([R1, R2])
L_own = 1e-7      # произвольное значение для диагонали

M_expected = M_analytic_coaxial(R1, R2, gap)
print(f"Аналитика (квазистатика): {M_expected * 1e9:.6f} нГн")

# f = 0
L0 = inductance_matrix_nonqs(delta, A, n, m, R, L_own, f=0.0, N=256)
print(f"Численно f=0:             {L0[0, 1].real * 1e9:.6f} нГн")
print(f"Относительная ошибка:     "
      f"{abs(L0[0,1].real - M_expected) / abs(M_expected):.3e}")

# f = 1 ГГц
L1 = inductance_matrix_nonqs(delta, A, n, m, R, L_own, f=1e9, N=256)
print(f"Численно f=1 ГГц:         {L1[0, 1].real * 1e9:.6f} "
      f"{L1[0, 1].imag * 1e9:+.6f}j нГн")

# Сравнение со квазистатикой
print(f"Изменение Re(M):          "
      f"{(L1[0,1].real - L0[0,1].real) * 1e9:+.6f} нГн")
print(f"Im(M)/Re(M) при 1 ГГц:    "
      f"{L1[0,1].imag / L1[0,1].real:.3e}")


# ===========================================================================
# ТЕСТ 2: два кольца в разных стопках (m=2, n=1)
# ===========================================================================
print("\n=== Тест 2: два кольца в разных стопках (m=2, n=1) ===")

A = 0.10
n = 1
m = 2
R = np.array([0.08])
delta = np.array([0.0])   # не используется при n=1, но должен быть корректным

L0_2 = inductance_matrix_nonqs(delta, A, n, m, R, L_own, f=0.0, N=256)
print(f"f=0 : M(0,1) = {L0_2[0, 1].real * 1e9:.6f} нГн")

L1_2 = inductance_matrix_nonqs(delta, A, n, m, R, L_own, f=1e9, N=256)
print(f"f=1ГГц: M(0,1) = {L1_2[0, 1].real * 1e9:.6f} "
      f"{L1_2[0, 1].imag * 1e9:+.6f}j нГн")
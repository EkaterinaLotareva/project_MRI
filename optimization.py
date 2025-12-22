import numpy as np
import main
from magnetic_field import *
from currents import *
from inductance import *



def uniformity(C, R, n, m, delta, phi, grid):
    "Рассчитывает меру отклонения от однородности поля (чем меньше значение, тем более однородное)"
    L = inductance_matrix(n, m, R, main.r, main.A, delta)
    Z_self = Z_self_matrix(main.rho, main.r, C, n, m, R)
    U = generate_voltage_array(main.U_0, m, n, phi)
    I = calc_I(Z_self, U, main.w, L, n, m)
    B = b_s_l(grid, I, main.N, n, m, points_on_rings_general(delta, n, main.A, main.N, R, m))
    uniformity = np.std(B) / np.abs(np.mean(B))


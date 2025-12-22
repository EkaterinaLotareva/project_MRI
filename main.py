from inductance import *
from currents import *
from magnetic_field import *
import math

R = [3.5, 3.5, 3.5, 3.5, 3.5] #см
Delta = [1, 0.1, 1, 0.1] #см
m = 4
n = 5
A = 8 #см
w = math.pi*2*63.8 #Мгц
U_0 = 1
C = None
r = 1 #радиус сечения кольца
rho = 1.724*(10**3)
resistance = []
N = 512

R_rel = [] #относительные размеры
delta = [] #относительные размеры
for i in range(len(R)):
    R_rel.append(R[i]/A)
for i in range(len(Delta)):
    delta.append(Delta[i]/A)

L = inductance_matrix(n, m, R_rel, r, A, delta)
Z_self = Z_self_matrix(rho, r, C, n, m, R_rel)
U = generate_voltage_array(U_0, m, n)
I = calc_I(Z_self, U, w, L, n, m)
obs_points = None

b_s_l(obs_points, I, N, n, m, points_on_rings_general(delta, n, A, N, R, m))

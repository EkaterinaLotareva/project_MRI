from geometry import SymmetricSystem
from inductance import *
from magnetic_field import *

R = [1, 1, 1, 1, 1]
m = 6
Delta = [1, 0.1, 1, 0.1]
n = 5
A = 8

r = [] #относительные размеры
delta = [] #относительные размеры
for i in range(len(R)):
    r.append(R[i]/A)
for i in range(len(Delta)):
    delta.append(Delta[i]/A)

MRI = SymmetricSystem(m, r, delta, n, A)

I_matrix = None
obs_points = None
b_s_l(obs_points, I_matrix, MRI.N, MRI.n, MRI.m, all_coordinates=MRI.points_on_rings_general())

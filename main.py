from geometry import SymmetricSystem
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
U = None
C = None
S = 1
rho = 1.724*(10**3)
resistance = []

for i in range(n):
    resistance.append(rho*2*math.pi*R[i]/S)
resistance = np.array(resistance)


r = [] #относительные размеры
delta = [] #относительные размеры
for i in range(len(R)):
    r.append(R[i]/A)
for i in range(len(Delta)):
    delta.append(Delta[i]/A)

MRI = SymmetricSystem(m, r, delta, n, A)

L = inductance_matrix(n, m, r, 1, delta)
I = currents(impedance(resistance, C, L, w), U)
obs_points = None
b_s_l(obs_points, I, MRI.N, MRI.n, MRI.m, all_coordinates=MRI.points_on_rings_general())

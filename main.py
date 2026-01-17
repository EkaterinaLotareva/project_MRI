from inductance import *
from currents import *
from magnetic_field import *
import math
from optimization import *
import numpy as np


R = [3.5, 3.5, 3.5, 3.5, 3.5] #см
Delta = [1, 0.1, 1, 0.1] #см
m = 4
n = 5
A = 8 #см
w = math.pi*2*63.8 #Мгц
U_0 = 1
C = [1]*(m*n)
r = 1 #радиус сечения кольца
rho = 1.724*(10**3)
resistance = []
N = 512
num_points = 1000

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

def grid(radius, num_points):
    side_points = int(np.sqrt(num_points / np.pi) * 2) + 1
    x = np.linspace(-radius, radius, side_points)
    y = np.linspace(-radius, radius, side_points)
    X, Y = np.meshgrid(x, y)
    distances = np.sqrt(X ** 2 + Y ** 2)
    mask = distances <= radius
    points_inside = np.column_stack((X[mask], Y[mask]))
    return points_inside

grid = grid(A, num_points)

#b_s_l(grid, I, N, n, m, points_on_rings_general(delta, n, A, N, R, m))

fixed_params = {'r': r, 'rho': rho, 'A': A, 'U_0': U_0, 'w': w, 'N': N, 'grid': grid}
#bounds = [(1e-9, 1e-6), (1, 10), (4, 20), (3, 10), (0.1, 10), (0, 2*np.pi)]
bounds = (
    np.array([1e-9, 0.1, 5, 2, 0.001, 0]),      # нижние границы (min)
    np.array([1e-6, 1.0, 50, 10, 0.1, 2*np.pi]) # верхние границы (max)
)
options = {
            'c1': 0.5,   # когнитивный параметр
            'c2': 0.3,   # социальный параметр
            'w': 0.9     # инерция
        }
optimizer = MRIOptimizer(fixed_params)
optimization = optimizer.optimize(bounds, options)
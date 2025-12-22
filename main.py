from geometry import SymmetricSystem
from inductance import *
from currents import *
from magnetic_field import *
import math

# Параметры системы
m = 2          # Количество стопок
rad = [0.035]  # Радиусы колец в стопке
deltas = []   # пока 1 кольцо в стопке
N_seg = 512 # мелкость разбиения колец
n = len(rad) # количство колец в стопке
sigma = 5.96e7 
r = np.full(shape=(m, n), fill_value=0.001)  # радиус сечения кольца
C = np.full(shape=(m, n), fill_value=0.001)  # емкость
omega = 63.8e6  # частота
x_off = rad[0]/2  # радиус внутренней зоны


R = [] #относительные размеры
delta = [] #относительные размеры
for i in range(len(R)):
    r.append(rad[i]/x_off)
for i in range(len(Delta)):
    delta.append(deltas[i]/x_off)


system = SymmetricSystem(m=m, R=rad, delta=deltas, N=N_seg, n=n, A=x_off)
all_coordinates = system.points_on_rings_general()


# Получение матрицы взаимных индуктивностей
L = inductance_matrix(n, m, rad, r, x_off, deltas)
# U = np.array([3.+0.j, 3.+0.j, 3.+0.j, 3.+0.j])
U = np.array(generate_U(2, m, n))
demo_Z = demo_Z_self_matrix(sigma, r, C, n, m, rad)

# Нахождение вектора токов
I = calc_I(demo_Z, U, omega, L)
I_matrix = I.reshape(m, n)


x_max = x_off + sum(deltas) + rad[-1]
x_min = -x_max
x_coords = np.linspace(x_min, x_max, 200)
y_coords = np.linspace(x_min, x_max, 200)

# Точки наблюдения по оси x P_obs = (x, 0, 0)
P_obs_X = np.stack([
    x_coords,
    np.zeros_like(x_coords),
    np.zeros_like(x_coords)
], axis=1)


Bmagx = b_s_l(obs_points=P_obs_X,
                 I_matrix=I_matrix,
                 N=N_seg,
                 n=n,
                 m=m,
                 all_coordinates=all_coordinates)


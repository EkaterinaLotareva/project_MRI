from inductance import *
from currents import *
from magnetic_field import *
import math
from optimization import *
import numpy as np


# Параметры системы
m = 6                      # Количество стопок (симметрия, разд. 7.4, Рис.2)
n = 5                      # Количество колец в стопке
rad = np.full(n, 0.035)    # Радиусы колец (массив длины n)
deltas = np.array([0.01, 0.001, 0.01, 0.001])  # Зазоры между кольцами (n-1 штук)
N_seg = 512                # Дискретизация кольца

# Физические параметры
r_ohm = 3.73e-4
L_own = 0.21e-6
C = np.full(n, 23.7e-12)   # Ёмкость конденсаторов (1D массив длины n)
omega = 2 * math.pi * 58.5e6  # Циклическая частота (рад/с)
U_0 = 1.0                  # Амплитуда напряжения
phi_0 = 0                 # Начальная фаза
x_off = 0.08               # Расстояние от центра до первого кольца 

sigma = 5.96e7           # Удельная проводимость меди (См/м)
rho = 1 / sigma          # Удельное сопротивление (Ом·м)


R_rel = [] #относительные размеры
delta = [] #относительные размеры
for i in range(len(R)):
    R_rel.append(R[i]/x_off)
for i in range(len(Delta)):
    delta.append(Delta[i]/x_off)
    
# Параметры области визуализации
k = 1.5
x_max = k * x_off

# Геометрия
print("Расчёт геометрии системы...")
all_coordinates = points_on_rings_general(delta, n, x_off, N_seg, R_real, m)

# Индуктивности
print("Расчёт матрицы индуктивностей...")
L = inductance_matrix(n, m, R_real, L_own, x_off, delta)

# Импедансы
print("Расчёт матрицы импедансов...")
Z_self = Z_self_matrix(rho, r_ohm, C, n, m, R_real omega)

# Напряжения
print("Генерация вектора напряжений...")
U = generate_voltage_array(U_0, m, n, phi_0)

# Токи
print("Расчёт токов...")
I_matrix = calc_I(Z_self, U, omega, L, n, m)

print(f"Токи в первой стопке (амплитуды): {np.abs(I_matrix[0, :])}")
print(f"Максимальный ток в системе: {np.max(np.abs(I_matrix)):.4f} А")

# Сетка наблюдения
print("Построение сетки наблюдения...")
x_coords = np.linspace(-x_max, x_max, 200)
y_coords = np.linspace(-x_max, x_max, 200)
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

# Точки наблюдения в плоскости z=0
obs_points = np.stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())], axis=1)

# Магнитное поле
print("Расчёт магнитного поля...")
B = b_s_l(obs_points, I_matrix, N_seg, n, m, all_coordinates)

# Амплитуда поля |B|
B_magnitude = np.linalg.norm(B, axis=1).reshape(X.shape)
B_magnitude_T = B_magnitude  # mu0 уже учтён в b_s_l

# Визуализация
print("Визуализация...")
fig, ax = plt.subplots(figsize=(8, 8))
contour = ax.contourf(X, Y, B_magnitude_T, levels=50, cmap='jet')
cbar = plt.colorbar(contour, ax=ax, label='|B| (Тл)')

# Положение колец в сечении z=0
for i_stack in range(m):
    for i_ring in range(n):
        center = ring_center_general(deltas, x_off, n, (2 * np.pi) / m, i_stack, i_ring)
        if np.abs(center[2]) < 0.01:  # Близко к плоскости z=0
            ax.plot(center[0], center[1], 'o', color='black', markersize=3, alpha=0.7)

ax.set_xlabel('x (м)')
ax.set_ylabel('y (м)')
ax.set_title(f'Стационарная картина магнитного поля |B|\n'
             f'ω = {omega/(2*np.pi)/1e6:.1f} МГц, m={m} стопок, n={n} колец')
ax.axis('equal')
ax.grid(True, linestyle=':', alpha=0.3)
ax.set_xlim(-x_max, x_max)
ax.set_ylim(-x_max, x_max)
plt.tight_layout()
plt.savefig('B_field_stationary.png', dpi=300)
plt.show()

# АЧХ
print("Построение резонансной характеристики...")
frequencies = np.linspace(50e6, 80e6, 50)
B_center = []
for f in frequencies:
    omega_test = 2 * np.pi * f
    Z_self_test = Z_self_matrix(rho, r_wire, C, n, m, rad, omega_test)
    I_test = calc_I(Z_self_test, U, omega_test, L, n, m)
    B_test = b_s_l(np.array([[0, 0, 0]]), I_test, N_seg, n, m, all_coordinates)
    B_center.append(np.linalg.norm(B_test[0]))


fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(frequencies/1e6, B_center, 'b-', linewidth=2)
ax2.axvline(omega/(2*np.pi)/1e6, color='red', linestyle='--',
           label=f'Рабочая частота = {omega/(2*np.pi)/1e6:.1f} МГц')
ax2.set_xlabel('Частота (МГц)')
ax2.set_ylabel('|B| в центре (Тл)')
ax2.set_title('Резонансная характеристика системы')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()
plt.tight_layout()
plt.savefig('resonance_curve.png', dpi=300)
plt.show()

# Оптимизация
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

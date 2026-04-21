import numpy as np
import math
import matplotlib.pyplot as plt

from src.geometry import points_on_rings_general, ring_center_general
from src.inductance import inductance_matrix
from src.currents import Z_self_matrix, generate_voltage_array, calc_I
from src.magnetic_field import b_s_l
from src.visualization import plot_field_contour
from src.optimization import MRIOptimizer  
from src.shoora import L_parallel  

# 1. ПАРАМЕТРЫ СИСТЕМЫ

m = 4                    # Количество стопок
n = 5                      # Количество колец в стопке
radii = np.full(n, 0.035)  # Радиусы колец (м)
# gaps = np.array([0.01])
gaps = np.array([0.01, 0.001, 0.01, 0.001])  # Зазоры между кольцами (n-1 штук)
# gaps = np.array([0.07])  # Зазоры между кольцами (n-1 штук)
N_seg = 512                # Дискретизация кольца (кол-во сегментов)

# Физические параметры
r_ohm = 3.73e-4            # Активное сопротивление кольца (Ом)
L_self = 0.21e-6            # Собственная индуктивность (Гн)
C = np.full(n, 23.7e-12)   # Ёмкость конденсаторов (Ф)
frequency_MHz = 71.294       # Частота (МГц)
omega = 2 * math.pi * frequency_MHz * 1e6  # Циклическая частота (рад/с)
U_0 = 200.0                  # Амплитуда напряжения (В)
phi_0 = 0                  # Начальная фаза
A = 0.08                   # Расстояние от центра до первого кольца (м)

# Параметры области визуализации
k = 2
x_max = k * A


# 2. РАСЧЁТ ГЕОМЕТРИИ

print("Расчёт геометрии системы...")
all_coordinates, normals = points_on_rings_general(
    delta=gaps,
    n=n,
    A=A,
    N=N_seg,
    R=radii,
    m=m
)

# Отладка: вывод центров и нормалей
fi = 2 * np.pi / m
print(f"\nПроверка центров и нормалей (m={m}, n={n}):")
print(f"Форма normals: {normals.shape}, тип: {type(normals)}")
print("-" * 80)
for stack_idx in range(m):
    angle = stack_idx * fi
    print(f"Стопка {stack_idx}: угол={angle:.4f} rad ({np.degrees(angle):.1f}°)")
    for ring_idx in range(n):
        center = ring_center_general(gaps, A, n, fi, stack_idx, ring_idx)
        normal = normals[stack_idx * n + ring_idx] if normals.shape[0] == m * n else normals[stack_idx]
        print(f"  Кольцо {ring_idx}: центр={np.round(center, 4)}, нормаль={np.round(normal, 4)}")
print("-" * 80)
print(f"Общее количество точек: {len(all_coordinates)}")

for i in range(m):
    for j in range(n):
        center = ring_center_general(delta=gaps, A=A, n=n, fi=2*np.pi/m, stack_index=i, ring_index=j)
        print(f"Центр кольца {j} в стопке {i}: {center}, нормаль: {normals[i]}")
        
# 3. РАСЧЁТ ИНДУКТИВНОСТЕЙ


print("Расчёт матрицы индуктивностей...")

L = inductance_matrix(
    n=n, m=m, R=radii, L_own=L_self, A=A, delta=gaps, 
    all_points=all_coordinates, normals=normals, N_seg=N_seg
)
print(f"Матрица индуктивности: {L}")

# 4. РАСЧЁТ ИМПЕДАНСОВ И ТОКОВ"""
print("Расчёт матрицы импедансов...")
Z_self = Z_self_matrix(
    r=r_ohm,           
    C=C,
    n=n,
    m=m,
    R=radii,
    omega=omega
)

# print(f"Матрица импедансов: {Z_self - 1j * omega * L}")

print("Генерация вектора напряжений...")
# U = np.full()
U = generate_voltage_array(U_0, m, n, phi_0)

print("Расчёт токов...")
I_matrix = calc_I(Z_self, U, omega, L, n, m)

print(f"Токи в первой стопке (амплитуды): {np.abs(I_matrix[0, :])}")
# print(f"Токи: {I_matrix}")

# Частотная зависимость амплитуд токов для всех стопок
print("Расчёт частотных характеристик токов для всех стопок...")
frequencies = np.linspace(20, 140, 100000)
current_amplitudes = np.zeros((frequencies.size, m, n))

for idx, f in enumerate(frequencies):
    if f == 0:
        current_amplitudes[idx, :, :] = 0.0
        continue
    omega_f = 2 * math.pi * f * 1e6
    Z_self_f = Z_self_matrix(
        r=r_ohm,
        C=C,
        n=n,
        m=m,
        R=radii,
        omega=omega_f
    )
    I_f = calc_I(Z_self_f, U, omega_f, L, n, m)
    current_amplitudes[idx, :, :] = np.abs(I_f)
 

fig, axes = plt.subplots(m, 1, figsize=(10, 3*m))
if m == 1:
    axes = [axes]

for stack_idx in range(m):
    ax = axes[stack_idx]
    ax.set_yscale('log')
    for ring_idx in range(n):
        ax.plot(frequencies, current_amplitudes[:, stack_idx, ring_idx],
                label=f'Кольцо {ring_idx + 1}')
    
    ax.set_xlabel('Частота, МГц')
    ax.set_ylabel('Амплитуда тока, А')
    ax.set_title(f'Амплитуды токов в кольцах стопки {stack_idx + 1} от частоты')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('stack1_current_amplitudes_vs_frequency.png', dpi=300)
plt.show()

# 5. РАСЧЁТ МАГНИТНОГО ПОЛЯ

print("Построение сетки наблюдения...")

x_coords = np.linspace(-x_max, x_max, 200)
y_coords = np.linspace(-x_max, x_max, 200)

"""
x_coords = np.linspace(0, 1.5*A + gaps[0], 200)
y_coords = np.linspace(-1.2*radii[0], 1.2*radii[0], 200) # для дыух колец
"""
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

# Точки наблюдения в плоскости z=0
obs_points = np.stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())], axis=1)

# Комплексное поле (векторное)
B_complex = b_s_l(obs_points, I_matrix, N_seg, n, m, all_coordinates, normals)

# Модуль вектора в каждой точке
B_amplitude = np.linalg.norm(B_complex, axis=1)

"""
min_distance = 0.01  # м
mask = np.zeros(obs_points.shape[0], dtype=bool)
fi = (2 * np.pi) / m
for i_stack in range(m):
    for i_ring in range(n):
        center = ring_center_general(gaps, A, n, fi, i_stack, i_ring)
        normal = normals[i_stack]
        rad = radii[i_ring]
        d = obs_points - center
        dist_to_plane = np.abs(np.dot(d, normal))
        proj = d - np.outer(np.dot(d, normal), normal)
        rho = np.linalg.norm(proj, axis=1)
        dist_to_ring = np.sqrt(dist_to_plane**2 + (rho - rad)**2)
        mask |= dist_to_ring < min_distance

valid = ~mask  # Точки, где расчёт физически корректен
if np.any(valid):
    idx = np.argmin(np.linalg.norm(obs_points[mask, None] - obs_points[valid], axis=2), axis=1)
    B_complex[mask] = B_complex[valid][idx]
    B_amplitude = np.linalg.norm(B_complex, axis=1)
"""


# 6. ВИЗУАЛИЗАЦИЯ
print("Визуализация магнитного поля...")


ring_centers = []
fi = (2 * np.pi) / m
for i_stack in range(m):
    for i_ring in range(n):
        center = ring_center_general(gaps, A, n, fi, i_stack, i_ring)
        if np.abs(center[2]) < 0.01:  # Близко к плоскости z=0
            ring_centers.append(center[:2])

plot_field_contour(
    X=X,
    Y=Y,
    B_magnitude=B_amplitude.reshape(X.shape),
    ring_centers=ring_centers,
    title=f'Стационарная картина магнитного поля |B|\nω = {frequency_MHz:.1f} МГц, m={m}, n={n}',
    save_path='B_field_stationary.png',
    norm_method='asinh'
)
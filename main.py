import numpy as np
import math
import matplotlib.pyplot as plt

from src.geometry import points_on_rings_general, ring_center_general
from src.inductance import inductance_matrix
from src.currents import Z_self_matrix, generate_voltage_array, calc_I
from src.magnetic_field import b_s_l
from src.visualization import plot_field_contour, plot_resonance_curve
from src.optimization import MRIOptimizer  

# 1. ПАРАМЕТРЫ СИСТЕМЫ

m = 2                     # Количество стопок
n = 1                      # Количество колец в стопке
radii = np.full(n, 0.035)  # Радиусы колец (м)
#gaps = np.array([0.01, 0.001, 0.01, 0.001])  # Зазоры между кольцами (n-1 штук)
gaps = np.array([0.07])  # Зазоры между кольцами (n-1 штук)
N_seg = 512                # Дискретизация кольца (кол-во сегментов)

# Физические параметры
r_ohm = 3.73e-4            # Активное сопротивление кольца (Ом)
L_self = 0.21e-6            # Собственная индуктивность (Гн)
C = np.full(n, 23.7e-12)   # Ёмкость конденсаторов (Ф)
frequency_MHz = 68.5       # Частота (МГц)
omega = 2 * math.pi * frequency_MHz * 1e6  # Циклическая частота (рад/с)
U_0 = 500.0                  # Амплитуда напряжения (В)
phi_0 = 0                  # Начальная фаза
A = 0.0175                   # Расстояние от центра до первого кольца (м)

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


# 3. РАСЧЁТ ИНДУКТИВНОСТЕЙ


print("Расчёт матрицы индуктивностей...")

L = inductance_matrix(
    n=n, m=m, R=radii, L_own=L_self, A=A, delta=gaps, 
    all_points=all_coordinates, normals=normals, N_seg=N_seg
)
print(f"Матрица индуктивности: {L}")

# 4. РАСЧЁТ ИМПЕДАНСОВ И ТОКОВ

print("Расчёт матрицы импедансов...")
Z_self = Z_self_matrix(
    r=r_ohm,           
    C=C,
    n=n,
    m=m,
    R=radii,
    omega=omega
)

print("Генерация вектора напряжений...")
# U = np.full()
U = generate_voltage_array(U_0, m, n, phi_0)

print("Расчёт токов...")
I_matrix = calc_I(Z_self, U, omega, L, n, m)

print(f"Токи в первой стопке (амплитуды): {np.abs(I_matrix[0, :])}")
print(f"Токи: {I_matrix}")

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

# 3. Если нужно построить график, используем B_amplitude

# 6. ВИЗУАЛИЗАЦИЯ
print("Визуализация магнитного поля...")

# Собираем центры колец для отображения на графике
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
    save_path='B_field_stationary.png'
)

# 7. ОПТИМИЗАЦИЯ
"""

# 1. Фиксированные параметры
fixed_params = {
    'r_ohm': 3.73e-4,           # Ом
    'A': 0.08,              # м
    'U_0': 500,             # В
    'N': 200,               # кол-во точек
    'grid': np.linspace(-0.2, 0.2, 200),  # сетка для оценки однородности
    'L_own': 0.21e-6,          # Гн
    'R': np.full(n, 0.035),    # м (радиус колец)
    'n': 5,                 # шт (кол-во колец в стопке)
    'm': 6,                 # шт (кол-во стопок)
    'delta': np.array([0.01, 0.001, 0.01, 0.001]),          # м (зазор между кольцами)
    'B_target': 10e-6,        # Целевая индукция (10 мкТл)
    'weight_magnitude': 0.3,  # Вес величины поля (0.3 = 30% важности)
    'B_min_threshold': 1e-6,
}

# 2. Создание оптимизатора
optimizer = MRIOptimizer(fixed_params=fixed_params)

# 3. Границы для варьируемых параметров [C, omega, phi]
bounds = (
    [10e-12,   2*np.pi*50e6,   0.0],        # минимумы: C=10пФ, f=50МГц, phi=0
    [100e-12,  2*np.pi*100e6,  2*np.pi]    # максимумы: C=100пФ, f=100МГц, phi=2π
)

# 4. Параметры PSO
pso_options = {
    'c1': 0.5,
    'c2': 0.3,
    'w': 0.9,
}

# 5. Запуск оптимизации
best_pos, best_cost = optimizer.optimize(
    bounds=bounds,
    pso_options=pso_options,
    n_particles=10,
    max_iterations=10
)

# 6. Результаты
print("\n=== Оптимальные параметры ===")
print(f"Ёмкость:     {best_pos[0]:.2e} Ф ({best_pos[0]*1e12:.2f} пФ)")
print(f"Частота:     {best_pos[1]/(2*np.pi):.2f} МГц (omega={best_pos[1]:.2e} рад/с)")
print(f"Фаза:        {best_pos[2]:.4f} рад ({np.degrees(best_pos[2]):.2f}°)")
print(f"Однородность: {best_cost:.6f}")


print("Построение резонансной характеристики...")
frequencies = np.linspace(50e6, 80e6, 50)
B_center = []

# Уменьшаем дискретизацию для ускорения расчёта АЧХ
N_seg_ACH = 128  

for i, f in enumerate(frequencies):
    omega_test = 2 * np.pi * f
    Z_self_test = Z_self_matrix(
        r=r_ohm, C=C, n=n, m=m, R=radii, omega=omega_test
    )
    I_test = calc_I(
        Z_self=Z_self_test, U=U, omega=omega_test, L=L, n=n, m=m
    )
    B_test = b_s_l(
        obs_points=np.array([[0, 0, 0]]),
        I_matrix=I_test,
        N=N_seg_ACH,
        n=n,
        m=m,
        all_coordinates=all_coordinates
    )
    B_center.append(np.linalg.norm(B_test[0]))
    
    # Прогресс
    if (i + 1) % 10 == 0:
        print(f"  Частоты обработаны: {i + 1}/{len(frequencies)}")

B_center = np.array(B_center)

# Проверка на пустые данные
if np.all(np.isnan(B_center)):
    print(" Предупреждение: Все значения поля NaN!")
else:
    plot_resonance_curve(
        frequencies=frequencies,
        B_values=B_center,
        resonance_freq=frequency_MHz * 1e6,  
        save_path='resonance_curve.png'
    )
"""
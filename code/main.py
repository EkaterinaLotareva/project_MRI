import numpy as np

S = 1.14563838e-02
k1, k2, k3 = 3.29859053e-02, 4.11406771e-02, 3.78943057e-02


# Распределяем общую сумму S между четырьмя дельтами
d1 = S * k1
d2 = (S - d1) * k2
d3 = (S - d1 - d2) * k3
d4 = S - d1 - d2 - d3

deltas = np.array([d1, d2, d3, d4])
print(sum(deltas))

import numpy as np
import os
import json
from datetime import datetime
import src.config as config  
import matplotlib.pyplot as plt
from src.geometry import points_on_rings_general, all_rings_centers
from src.inductance import inductance_matrix
from src.currents import Z_self_matrix, generate_voltage_array, calc_I
from src.magnetic_field import b_s_l_optimized, quality_metric
from src.visualization import plot_field_contour, plot_field_along_axis, visualize_rings
from src.utils import create_run_directory, save_simulation_data

print("=== MRI Field Simulation ===")


# 1. Загрузка параметров
m, n, radii, gaps = config.m, config.n, config.radii, config.gaps
A, N_seg, omega, r_ohm, U_0, C = config.A, config.N_seg, config.omega, config.r_ohm, config.U_0, config.C

# параметры до оптимиации
gaps_0 = np.array([0.01, 0.001, 0.01, 0.001])
C_0 = 

# 2. Расчет геометрии и индуктивностей
all_coords, normals = points_on_rings_general(delta=gaps, n=n, A=A, N=N_seg, R=radii, m=m)
L = inductance_matrix(n=n, m=m, R=radii, L_own=config.L_self, A=A, delta=gaps, normals=normals)

# 3. Расчет матриц токов
Z_self = Z_self_matrix(r=r_ohm, C=C, n=n, m=m, R=radii, omega=omega)
U = generate_voltage_array(U_0=U_0, m=m, n=n)
I_flat = calc_I(Z_self=Z_self, L=L, omega=omega, U=U, n=n, m=m)
I_matrix = I_flat.reshape((m, n))

ring_centers = all_rings_centers(gaps, A, n, m, 2*np.pi / m)

# Цветовая карта
phi = np.linspace(0, 2 * np.pi, 100)
r = np.linspace(0, 0.6 * config.A, 100)
R, Phi = np.meshgrid(r, phi, indexing='ij')

X = R * np.cos(Phi)
Y = R * np.sin(Phi)
obs_points_2d = np.stack((X.ravel(), Y.ravel(), np.zeros_like(X.ravel())), axis=1)
B_2d_complex = b_s_l_optimized(obs_points_2d, I_matrix, n, m, ring_centers, normals, radii)
B_2d_amp = np.linalg.norm(B_2d_complex, axis=1).reshape(X.shape)

plot_field_contour(
    Phi=Phi, R=R, B_magnitude=B_2d_amp, C=C,
    title=f'B amplitude, (C = {C[0] * 1e12:.1f} pF,f = {config.frequency_MHz} МГц)',
    save_path='B_field_contour.png'
)

print("Цветовая карта сохранена в 'B_field_contour.png'")

# График амплитуды вдоль оси X
print("Расчет поля вдоль оси X...")

x_line = np.linspace(-0.05, 0.05, 200) 
obs_points_1d_x = np.stack((x_line, np.zeros_like(x_line), np.zeros_like(x_line)), axis=1)

B_1d_complex_x = b_s_l_optimized(obs_points_1d_x, I_matrix, n, m, ring_centers, normals, radii)
B_1d_amp_x = np.linalg.norm(B_1d_complex_x, axis=1)

plot_field_along_axis(
    axis_coords=x_line, B_amplitude=B_1d_amp_x, axis_name='X',
    title='Профиль амплитуды поля |B| вдоль оси X',
    save_path='B_field_along_X.png'
)
print("График вдоль оси X сохранен в 'B_field_along_X.png'")
print("Расчет поля вдоль оси Y...")

y_line = np.linspace(-0.05, 0.05, 200)
obs_points_1d_y = np.stack((np.zeros_like(y_line), y_line, np.zeros_like(y_line)), axis=1)

B_1d_complex_y = b_s_l_optimized(obs_points_1d_y, I_matrix, n, m, ring_centers, normals, radii)
B_1d_amp_y = np.linalg.norm(B_1d_complex_y, axis=1)

print("График вдоль оси Y сохранен в 'B_field_along_Y.png'")
print(f"Значение целевой функции: {quality_metric(I_matrix, n, m, ring_centers, normals, radii, obs_points_1d_y):.6e}")

centers = all_rings_centers(
    gaps, A, n, m, 2*np.pi / m
)

visualize_rings(
    all_coords,
    ring_centers,
    normals,
    N_seg,
    n,
    m
)

current_config = {
    'frequency_MHz': 68.5,
    'n': n,
    'm': m,
    'A': A, 
    'C': C,
    'gaps': gaps,
    'Значение целевой функции': quality_metric(I_matrix, n, m, ring_centers, normals, radii, obs_points=obs_points_1d_y) 
    #добавить сюда оптимизированные параметры
}

matrices_to_save = {
    'L_matrix': L,
    'I_vector': I_matrix,
    'B_2d_amplitude': B_2d_amp,
    'X_grid': X,
    'Y_grid': Y
}

# Шаг 1: Создаем уникальную папку
run_folder = create_run_directory()

# Шаг 2: Сохраняем конфиг и массивы
save_simulation_data(run_folder, current_config, matrices_to_save)

# Шаг 3: Передаем этот же путь в функции отрисовки графиков
plot_field_contour(
    Phi=Phi, R=R, B_magnitude=B_2d_amp, C=C,
    title=f'B amplitude, (C = {C[0] * 1e12:.1f} pF,f = {config.frequency_MHz} МГц)',
    save_path=os.path.join(run_folder, "B_field_contour.png")
)

plot_field_along_axis(
    axis_coords=x_line, B_amplitude=B_1d_amp_x, axis_name='X',
    title='Профиль амплитуды поля |B| вдоль оси X',
    save_path=os.path.join(run_folder, "B_field_along_X.png")
)

plot_field_along_axis(
    axis_coords=y_line, B_amplitude=B_1d_amp_y, axis_name='Y',
    title='Профиль амплитуды поля |B| вдоль оси Y',
    save_path=os.path.join(run_folder, "B_field_along_Y.png")
)

print(f"Все графики сохранены в папку: {run_folder}")

import numpy as np
import matplotlib.pyplot as plt


# 1. Ваши исходные данные
points = np.array([
    [8.881784197001252e-16, 0.011133333333333335],
    [7.028301886792454, 0.011122222222222224],
    [8.018867924528303, 0.009044444444444445],
    [15, 0.009044444444444445],
    [15.99056603773585, 0.005722222222222222],
    [28.915094339622637, 0.005711111111111111]
])

# 2. Интерполяция (генерация 100 точек на сегмент)
def interpolate_polyline(points, num_points_per_segment=100):
    all_points = []
    for i in range(len(points) - 1):
        t = np.linspace(0, 1, num_points_per_segment)
        segment = points[i] + t[:, np.newaxis] * (points[i+1] - points[i])
        all_points.append(segment[:-1])
    all_points.append([points[-1]])
    return np.vstack(all_points)

new_data = interpolate_polyline(points, 100)
x = new_data[:, 0]
y = new_data[:, 1]

# 3. Построение графика
plt.figure(figsize=(10, 6))

# Используем semilogy для логарифмической шкалы по оси Y
plt.semilogy(x, y, label='Ломаная (log Y)', color='blue', linewidth=2)

plt.title('График с логарифмической шкалой по оси Y')
plt.xlabel('X (линейный масштаб)')
plt.ylabel('Y (log scale)')
plt.grid(True, which="both", ls="-", alpha=0.5) # which="both" показывает и основные и промежуточные линии сетки
plt.legend()

plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 1. Импортируем функции твоего физического движка из src
from src.geometry import points_on_rings_general, all_rings_centers
from src.inductance import inductance_matrix
from src.currents import Z_self_matrix, generate_voltage_array, calc_I
from src.magnetic_field import b_s_l_optimized


# ==========================================
# ЧАСТЬ 1: ФУНКЦИИ (ПОЛНОСТЬЮ БЕЗ КЛАССОВ)
# ==========================================

def compute_b_field_clean(position, grid_points, n, m, omega, A, U_0, r_ohm, R_array, phi, L_own):
    """
    Чистая функция: принимает только необходимые физические данные, 
    никаких объектов класса внутри.
    """
    C = position[0]
    deltas = position[1:5]
    
    # Геометрия системы
    all_coords, normals = points_on_rings_general(
        delta=deltas, n=n, A=A, N=512, R=R_array, m=m
    )
    L_new = inductance_matrix(
        n=n, m=m, R=R_array, L_own=L_own, A=A, delta=deltas, normals=normals
    )
    centers = all_rings_centers(deltas, A, n, m, phi)
    
    # Токи
    C_array = np.full(n * m, C)
    Z_self = Z_self_matrix(r=r_ohm, C=C_array, n=n, m=m, R=R_array, omega=omega)
    U = generate_voltage_array(U_0, m, n, phi)
    I = calc_I(Z_self, U, omega, L_new, n, m)
    
    # Расчет комплексного поля и взятие его модуля (амплитуды)
    B_complex = b_s_l_optimized(grid_points, I, n, m, centers, normals, R_array)
    return np.linalg.norm(B_complex, axis=1)


def plot_field_contour_updated(Phi, R, B_magnitude, C, title="", save_path="B_field.png", vmin=None, vmax=None):
    """Отрисовка цветовой карты магнитного поля в полярных координатах"""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    # ЕСЛИ переданы общие границы, создаем фиксированный массив уровней
    if vmin is not None and vmax is not None:
        levels = np.linspace(vmin, vmax, 200)  # 200 уровней дают идеально гладкий градиент
    else:
        levels = 1000  # дефолтное поведение автошкалирования
    
    contour = ax.contourf(Phi, R, B_magnitude, levels=levels,
                          cmap='jet',  
                          extend='both')
    
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('|B| (Тл)', rotation=270, labelpad=20)
    
    # Оформление полярной сетки
    full_title = f"{title}\nB amplitude, C: {C[0] * 1e12:.1f} pF" if title else f"B amplitude, C: {C[0] * 1e12:.1f} pF"
    ax.set_title(full_title, pad=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.set_rlabel_position(-22.5) 
    
    # Сохранение и закрытие фигуры
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()  # Чтобы сразу увидеть в Jupyter Notebook
    plt.close()


# ==========================================
# ЧАСТЬ 2: РАСПАКОВКА НАСТРОЕК И СЕТКА
# ==========================================

# Вытаскиваем "сырые" переменные из твоего словаря fixed_params, заданного в начале ноутбука
n = 5
m = 5
omega = 2 * np.pi * 68.5 * 1e6
phi = 0
U_0 = 200
r_ohm = 3.73e-4 
A = 0.8
L_own = 1.5e-7
R_domain = 0.6 * 0.08

R_array = np.full(n, 0.035) 

# Создаем двумерную полярную расчетную сетку (Phi, R)
N_r = 100
N_phi = 120
r_vec = np.linspace(0, R_domain, N_r)
phi_vec = np.linspace(0, 2 * np.pi, N_phi)
Phi, R = np.meshgrid(phi_vec, r_vec)

# Переводим полярные координаты сетки в декартовы (X, Y, Z=0) для физического расчета
X_flat = (R * np.cos(Phi)).ravel()
Y_flat = (R * np.sin(Phi)).ravel()
Z_flat = np.zeros_like(X_flat)
grid_points_polar = np.stack((X_flat, Y_flat, Z_flat), axis=1)


# ==========================================
# ЧАСТЬ 3: РАСЧЕТ И ОТРИСОВКА (ДО / ПОСЛЕ)
# ==========================================

# Гарантируем, что массивы позиций плоские (1D векторы длиной 5)
pos_before = [2.37e-1, 0.01, 0.001, 0.01, 0.001] 
pos_after = [1.12880619e-11, 2.24878429e-03, 3.96943456e-03, 3.48331688e-03,
 1.76855286e-02]

print("Считаем поле ДО оптимизации...")
B_before_flat = compute_b_field_clean(
    pos_before, grid_points_polar, 
    n, m, omega, A, U_0, r_ohm, R_array, phi, L_own
)

print("Считаем поле ПОСЛЕ оптимизации...")
B_after_flat = compute_b_field_clean(
    pos_after, grid_points_polar, 
    n, m, omega, A, U_0, r_ohm, R_array, phi, L_own
)

# Возвращаем массивам полей форму исходной сетки (N_r, N_phi)
B_before = B_before_flat.reshape(N_r, N_phi)
B_after = B_after_flat.reshape(N_r, N_phi)

# Находим глобальные минимумы и максимумы для ЖЕСТКОЙ фиксации шкал
vmin = min(B_before.min(), B_after.min())
vmax = max(B_before.max(), B_after.max())

print(f"\nВычислен общий диапазон поля: vmin = {vmin:.3e} Тл, vmax = {vmax:.3e} Тл")
print("Строим карты с одинаковой шкалой...")

# Строим первую карту (ДО)
plot_field_contour_updated(
    Phi, R, B_magnitude=B_before, C=pos_before, 
    title="Поле ДО оптимизации", 
    save_path="B_field_before.png", 
    vmin=vmin, vmax=vmax
)

# Строим вторую карту (ПОСЛЕ)
plot_field_contour_updated(
    Phi, R, B_magnitude=B_after, C=pos_after, 
    title="Поле ПОСЛЕ оптимизации (PSO)", 
    save_path="B_field_after.png", 
    vmin=vmin, vmax=vmax
)
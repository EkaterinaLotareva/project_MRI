import os
import numpy as np
import src.config as config  
from src.geometry import points_on_rings_general, all_rings_centers
from src.inductance import inductance_matrix
from src.currents import Z_self_matrix, generate_voltage_array, calc_I
from src.magnetic_field import b_s_l_optimized, quality_metric
from src.visualization import plot_field_contour, plot_field_along_axis, visualize_rings
from src.utils import create_run_directory, save_simulation_data

print("=== MRI Field Simulation ===")

# 1. Параметры
m, n, radii, gaps = config.m, config.n, config.radii, config.gaps
A, N_seg, omega, r_ohm, U_0, C = config.A, config.N_seg, config.omega, config.r_ohm, config.U_0, config.C
fi = 2 * np.pi / m

# 2. Геометрия и индуктивности
all_coords, normals = points_on_rings_general(delta=gaps, n=n, A=A, N=N_seg, R=radii, m=m)
L = inductance_matrix(n=n, m=m, R=radii, L_own=config.L_self, A=A, delta=gaps)
ring_centers = all_rings_centers(gaps, A, n, m, fi)

# 3. Токи
Z_self = Z_self_matrix(r=r_ohm, C=C, n=n, m=m, R=radii, omega=omega)
U = generate_voltage_array(U_0=U_0, m=m, n=n)
I_matrix = calc_I(Z_self=Z_self, U=U, omega=omega, L=L, n=n, m=m)

# 4. Расчет поля на 2D-сетке
phi_grid = np.linspace(0, 2 * np.pi, 100)
r_grid = np.linspace(0, 0.6 * config.A, 100)
R, Phi = np.meshgrid(r_grid, phi_grid, indexing='ij')

X = R * np.cos(Phi)
Y = R * np.sin(Phi)
obs_points_2d = np.stack((X.ravel(), Y.ravel(), np.zeros_like(X.ravel())), axis=1)

B_2d_complex = b_s_l_optimized(obs_points_2d, I_matrix, n, m, ring_centers, normals, radii)
B_2d_amp = np.linalg.norm(B_2d_complex, axis=1).reshape(X.shape)

# 5. Одномерные профили поля
x_line = np.linspace(-0.05, 0.05, 200) 
obs_points_1d_x = np.stack((x_line, np.zeros_like(x_line), np.zeros_like(x_line)), axis=1)
B_1d_amp_x = np.linalg.norm(b_s_l_optimized(obs_points_1d_x, I_matrix, n, m, ring_centers, normals, radii), axis=1)

y_line = np.linspace(-0.05, 0.05, 200)
obs_points_1d_y = np.stack((np.zeros_like(y_line), y_line, np.zeros_like(y_line)), axis=1)
B_1d_amp_y = np.linalg.norm(b_s_l_optimized(obs_points_1d_y, I_matrix, n, m, ring_centers, normals, radii), axis=1)

# Оценка однородности
metric_val = quality_metric(I_matrix, n, m, ring_centers, normals, radii, R_domain=config.R_domain)
print(f"Значение целевой функции (неоднородность): {metric_val:.6e}")

# Визуализация и сохранение
visualize_rings(all_coords, ring_centers, normals, N_seg, n, m)

run_folder = create_run_directory()
plot_field_contour(Phi, R, B_2d_amp, C=C, title="Карта амплитуды |B|", save_path=os.path.join(run_folder, "B_field_contour.png"))
plot_field_along_axis(x_line, B_1d_amp_x, axis_name='X', title='Профиль |B| вдоль X', save_path=os.path.join(run_folder, "B_field_along_X.png"))
plot_field_along_axis(y_line, B_1d_amp_y, axis_name='Y', title='Профиль |B| вдоль Y', save_path=os.path.join(run_folder, "B_field_along_Y.png"))

print(f"Результаты сохранены в: {run_folder}")
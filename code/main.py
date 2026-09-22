import os
import numpy as np
import src.config as config  
from src.geometry import points_on_rings_general, all_rings_centers, stack_basis
from src.inductance_quasystat import inductance_matrix
from src.nonqs_inductance import inductance_matrix_nonqs
from src.currents import Z_self_matrix, generate_voltage_array, calc_I
from src.magnetic_field import b_s_l_optimized, quality_metric
from src.visualization import plot_field_contour, plot_field_along_axis, visualize_rings, plot_field_comparison
from src.utils import create_run_directory, save_simulation_data

print("=== MRI Field Simulation ===")

# ==========================================
# 0. НАСТРОЙКА РЕЖИМА РАСЧЕТА
# ==========================================
# Поменяйте на 'qs' для быстрого квазистатического расчета
# Поменяйте на 'nonqs' для расчета с учетом запаздывания
CALC_MODE = 'both'  # 'qs', 'nonqs', или 'both' для сравнения
COMPARE_MODES = True  # Построить сравнительный график

# 1. Параметры
m, n, radii, gaps = config.m, config.n, config.radii, config.gaps
A, N_seg, omega, r_ohm, U_0, C, f_hz = config.A, config.N_seg, config.omega, config.r_ohm, config.U_0, config.C, config.frequency_MHz * 1e6
fi = 2 * np.pi / m

# 2. Геометрия и индуктивности
all_coords, normals = points_on_rings_general(delta=gaps, n=n, A=A, N=N_seg, R=radii, m=m)
ring_centers = all_rings_centers(gaps, A, n, m, fi)
e1, e2 = stack_basis(m)

# ==========================================
# 3. РАСЧЕТ ДЛЯ ОБОИХ РЕЖИМОВ
# ==========================================
results = {}

# Квазистатический режим
if CALC_MODE in ['qs', 'both']:
    print("\n[1/2] Расчет квазистатического режима...")
    L_qs = inductance_matrix(n=n, m=m, R=radii, L_own=config.L_self, A=A, delta=gaps)
    
    Z_self_qs = Z_self_matrix(r=r_ohm, C=C, n=n, m=m, R=radii, omega=omega)
    U_qs = generate_voltage_array(U_0=U_0, m=m, n=n)
    I_matrix_qs = calc_I(Z_self=Z_self_qs, U=U_qs, omega=omega, L=L_qs, n=n, m=m)
    
    # Профили поля
    x_line = np.linspace(-0.05, 0.05, 200)
    obs_points_1d_x = np.stack((x_line, np.zeros_like(x_line), np.zeros_like(x_line)), axis=1)
    B_1d_amp_x_qs = np.linalg.norm(
        b_s_l_optimized(obs_points_1d_x, I_matrix_qs, n, m, ring_centers, normals, radii), 
        axis=1
    )
    
    results['qs'] = {
        'L': L_qs,
        'I_matrix': I_matrix_qs,
        'B_x': B_1d_amp_x_qs,
    }
    print(f"  Максимум |B| (QS): {np.max(B_1d_amp_x_qs):.4e} Тл")

# Неквазистатический режим
if CALC_MODE in ['nonqs', 'both']:
    print("\n[2/2] Расчет неквазистатического режима...")
    L_nonqs = inductance_matrix_nonqs(
        ring_centers=ring_centers,
        normals=normals,
        e1=e1,
        e2=e2,
        radii=radii,
        n=n,
        m=m,
        L_own=config.L_self,
        f=f_hz,
        epsabs=1e-10,
        epsrel=1e-8
    )
    
    Z_self_nonqs = Z_self_matrix(r=r_ohm, C=C, n=n, m=m, R=radii, omega=omega)
    U_nonqs = generate_voltage_array(U_0=U_0, m=m, n=n)
    I_matrix_nonqs = calc_I(Z_self=Z_self_nonqs, U=U_nonqs, omega=omega, L=L_nonqs, n=n, m=m)
    
    # Профили поля
    x_line = np.linspace(-0.05, 0.05, 200)
    obs_points_1d_x = np.stack((x_line, np.zeros_like(x_line), np.zeros_like(x_line)), axis=1)
    B_1d_amp_x_nonqs = np.linalg.norm(
        b_s_l_optimized(obs_points_1d_x, I_matrix_nonqs, n, m, ring_centers, normals, radii), 
        axis=1
    )
    
    results['nonqs'] = {
        'L': L_nonqs,
        'I_matrix': I_matrix_nonqs,
        'B_x': B_1d_amp_x_nonqs,
    }
    print(f"  Максимум |B| (Non-QS): {np.max(B_1d_amp_x_nonqs):.4e} Тл")

# ==========================================
# 4. СРАВНИТЕЛЬНЫЙ АНАЛИЗ
# ==========================================
if COMPARE_MODES and 'qs' in results and 'nonqs' in results:
    print("\n=== СРАВНЕНИЕ РЕЖИМОВ ===")
    B_max_qs = np.max(results['qs']['B_x'])
    B_max_nonqs = np.max(results['nonqs']['B_x'])
    rel_diff = (B_max_nonqs - B_max_qs) / B_max_qs * 100
    
    print(f"Максимум |B| (QS):     {B_max_qs:.6e} Тл")
    print(f"Максимум |B| (Non-QS): {B_max_nonqs:.6e} Тл")
    
    # Построение сравнительного графика
    plot_field_comparison(
    x_line=x_line,
    B_amp_qs=B_1d_amp_x_qs,
    B_amp_nonqs=B_1d_amp_x_nonqs,
    axis_name='X',
    title=f'Профиль |B| вдоль X (f = {f_hz/1e6:.1f} МГц)',
    save_path='results/comparison_plot.png'
)

# ==========================================
# 5. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ==========================================
run_folder = create_run_directory(mode=CALC_MODE, frequency=f_hz if CALC_MODE != 'qs' else None)

if 'qs' in results:
    np.save(os.path.join(run_folder, "I_matrix_qs.npy"), results['qs']['I_matrix'])
    np.save(os.path.join(run_folder, "B_profile_x_qs.npy"), results['qs']['B_x'])
    
if 'nonqs' in results:
    np.save(os.path.join(run_folder, "I_matrix_nonqs.npy"), results['nonqs']['I_matrix'])
    np.save(os.path.join(run_folder, "B_profile_x_nonqs.npy"), results['nonqs']['B_x'])

print(f"\nРезультаты сохранены в: {run_folder}")
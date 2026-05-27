import numpy as np
import src.config as config  

from src.geometry import points_on_rings_general, ring_center_general
from src.inductance import inductance_matrix
from src.currents import Z_self_matrix, generate_voltage_array, calc_I
from src.magnetic_field import b_s_l_optimized
from src.visualization import plot_field_contour, plot_field_along_axis

print("=== MRI Field Simulation ===")

def main():
    print("Запуск стационарного расчета магнитного поля...")
    
    # 1. Загрузка параметров
    m, n, radii, gaps = config.m, config.n, config.radii, config.gaps
    A, N_seg, omega, r_ohm, U_0 = config.A, config.N_seg, config.omega, config.r_ohm, config.U_0
    
    # 2. Расчет геометрии и индуктивностей
    all_coords, normals = points_on_rings_general(delta=gaps, n=n, A=A, N=N_seg, R=radii, m=m)
    L = inductance_matrix(n=n, m=m, R=radii, L_own=config.L_self, A=A, delta=gaps, all_points=all_coords, normals=normals, N_seg=N_seg)
    
    # 3. Расчет матриц токов
    Z_self = Z_self_matrix(r=r_ohm, C=np.full(n*m, 3.07e-11), n=n, m=m, R=radii, omega=omega)
    U = generate_voltage_array(U_0=U_0, m=m, n=n)
    I_matrix = calc_I(Z_self=Z_self, U=U, omega=omega, L=L, n=n, m=m)
    
    # Массив центров колец для расчетов
    fi = (2 * np.pi) / m
    ring_centers = np.array([ring_center_general(gaps, A, n, fi, s, r) for s in range(m) for r in range(n)])

    # -------------------------------------------------------------
    # ПОСТРОЕНИЕ ЦВЕТОВОЙ КАРТЫ 
    # -------------------------------------------------------------
    print("Генерация сетки для цветовой карты...")
    grid_res = 100
    lim = 0.8 * A
    x_vals = np.linspace(-lim, lim, grid_res)
    y_vals = np.linspace(-lim, lim, grid_res)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Точки наблюдения в плоскости Z=0
    obs_points_2d = np.stack((X.ravel(), Y.ravel(), np.zeros_like(X.ravel())), axis=1)
    B_2d_complex = b_s_l_optimized(obs_points_2d, I_matrix, n, m, ring_centers, normals, radii)
    B_2d_amp = np.linalg.norm(B_2d_complex, axis=1).reshape(X.shape)
    
    plot_field_contour(
        X=X, Y=Y, B_magnitude=B_2d_amp, ring_centers=ring_centers,
        title=f'Распределение магнитного поля |B| (f = {config.frequency_MHz} МГц)',
        save_path='B_field_contour.png'
    )
    print("Цветовая карта сохранена в 'B_field_contour.png'")

    # -------------------------------------------------------------
    # ПОСТРОЕНИЕ ГРАФИКА ВДОЛЬ ОСИ X (1D)
    # -------------------------------------------------------------
    print("Расчет поля вдоль оси X...")
    x_line = np.linspace(-0.05, 0.05, 200)  # диапазон координат по X от -5см до 5см
    obs_points_1d = np.stack((x_line, np.zeros_like(x_line), np.zeros_like(x_line)), axis=1)
    
    B_1d_complex = b_s_l_optimized(obs_points_1d, I_matrix, n, m, ring_centers, normals, radii)
    B_1d_amp = np.linalg.norm(B_1d_complex, axis=1)
    
    plot_field_along_axis(
        axis_coords=x_line, B_amplitude=B_1d_amp, axis_name='X',
        title='Профиль амплитуды поля |B| вдоль оси X',
        save_path='B_field_along_X.png'
    )
    print("График вдоль оси X сохранен в 'B_field_along_X.png'")

if __name__ == '__main__':
    main()
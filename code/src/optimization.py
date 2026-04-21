import numpy as np
import pyswarms as ps
# Убедитесь, что пути импорта верные для вашего проекта
from src.geometry import points_on_rings_general
from src.magnetic_field import b_s_l
from src.currents import Z_self_matrix, generate_voltage_array, calc_I
from src.inductance import inductance_matrix

class MRIOptimizer:

    def __init__(self, fixed_params):
        self.fixed_params = fixed_params

    def uniformity(self, C, omega, phi):
        """Рассчитывает меру отклонения от однородности поля."""
        fp = self.fixed_params
        
        # 1. Исправлено: получаем N из словаря, а не несуществующую N_seg
        N = fp['N'] 
        r_ohm = fp['r_ohm']
        A = fp['A']
        U_0 = fp['U_0']
        grid = fp['grid']
        L_own = fp['L_own']
        R = fp['R']
        n = int(fp['n'])
        m = int(fp['m'])
        delta = fp['delta']

        if n <= 0 or m <= 0:
            return np.inf

        # Радиусы
        if np.isscalar(R):
            R_array = np.full(n, R)
        else:
            R_array = np.asarray(R, dtype=float)
            if R_array.size != n:
                return np.inf

        # Зазоры
        if np.isscalar(delta):
            delta_array = np.full(max(0, n - 1), delta)
        else:
            delta_array = np.asarray(delta, dtype=float)
            if delta_array.size != max(0, n - 1):
                return np.inf

        # Ёмкости
        if np.isscalar(C):
            C_array = np.full(n * m, C)
        else:
            C_array = np.asarray(C, dtype=float)
            if C_array.size == n:
                C_array = np.tile(C_array, m)
            elif C_array.size != n * m:
                return np.inf

        all_coordinates, normals = points_on_rings_general(
            delta=delta_array,
            n=n,
            A=A,
            N_seg=N,          
            R=R_array,        
            m=m
        )

        L = inductance_matrix(
            n=n, m=m, R=R_array, L_own=L_own, A=A, delta=delta_array, 
            all_points=all_coordinates, normals=normals, N_seg=N
        )
        
        # 3. Исправлено: убран дубль, используется C_array
        Z_self = Z_self_matrix(r=r_ohm, C=C_array, n=n, m=m, R=R_array, omega=omega)
        
        U = generate_voltage_array(U_0, m, n, phi)
        I = calc_I(Z_self, U, omega, L, n, m)

        # Сетка наблюдения
        grid_arr = np.asarray(grid)
        if grid_arr.ndim == 1:
            xx, yy = np.meshgrid(grid_arr, grid_arr, indexing='ij')
            obs_points = np.stack([xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())], axis=1)
        elif grid_arr.ndim == 2 and grid_arr.shape[1] == 2:
            obs_points = np.hstack([grid_arr, np.zeros((grid_arr.shape[0], 1))])
        elif grid_arr.ndim == 2 and grid_arr.shape[1] == 3:
            obs_points = grid_arr
        else:
            return np.inf

        # 4. Исправлено: добавлен аргумент normals, исправлено имя all_coordinates
        try:
            B = b_s_l(obs_points, I, N, n, m, all_coordinates, normals)
        except Exception:
            return np.inf

        B_magnitude = np.linalg.norm(B, axis=1)
        mean_B = np.mean(B_magnitude)
        
        if mean_B <= 0:
            return np.inf
            
        std_B = np.std(B_magnitude)
        B_target = fp.get('B_target', 1e-5)

        # Метрика качества
        quality_metric = (std_B + 1e-20) / (mean_B + 1e-20) * (B_target / (mean_B + 1e-20))

        return quality_metric

    def objective_function(self, positions: np.ndarray) -> np.ndarray:
        """Целевая функция для pyswarms."""
        n_particles = positions.shape[0]
        costs = np.zeros(n_particles)

        for i in range(n_particles):
            C = positions[i, 0]
            omega = positions[i, 1]
            phi = positions[i, 2]

            if C <= 0 or omega <= 0:
                costs[i] = np.inf
                continue

            try:
                costs[i] = self.uniformity(C, omega, phi)
            except Exception:
                costs[i] = np.inf

        return costs

    def optimize(self, bounds, pso_options, n_particles: int = 30, max_iterations: int = 50):
        """Запуск оптимизации."""
        if pso_options is None:
            pso_options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=3, 
            options=pso_options,
            bounds=bounds
        )

        best_cost, best_pos = optimizer.optimize(
            self.objective_function,
            iters=max_iterations,
            verbose=True
        )

        print(f"Лучшее значение функции потерь: {best_cost}")
        print(f"Лучшие параметры:")
        print(f'\n C = {best_pos[0]:.2e} Ф')
        print(f'\n omega = {best_pos[1]:.2e} рад/с')
        print(f'\n phi = {best_pos[2]:.4f} рад')

        return best_pos, best_cost
import numpy as np
from src.geometry import points_on_rings_general
from src.magnetic_field import *
from src.currents import *
from src.inductance import *
import pyswarms as ps


class MRIOptimizer:

    def __init__(self, fixed_params):
        self.fixed_params = fixed_params

    def uniformity(self, C, omega, phi):
        """Рассчитывает меру отклонения от однородности поля.

        Метрика: std(|B|) / |mean(|B|)|, чем меньше - тем лучше.
        """
        r_ohm = self.fixed_params['r_ohm']
        A = self.fixed_params['A']
        U_0 = self.fixed_params['U_0']
        N = self.fixed_params['N']
        grid = self.fixed_params['grid']
        L_own = self.fixed_params['L_own']

        R = self.fixed_params['R']
        n = int(self.fixed_params['n'])
        m = int(self.fixed_params['m'])
        delta = self.fixed_params['delta']

        if n <= 0 or m <= 0:
            raise ValueError("n и m должны быть положительными целыми")

        # Радиусы (в одной стопке), равные для всех m стопок
        if np.isscalar(R):
            R_array = np.full(n, R)
        else:
            R_array = np.asarray(R, dtype=float)
            if R_array.size != n:
                raise ValueError(f"Размер массива R ({R_array.size}) должен быть равен n ({n})")

        # Расставляем зазоры между n кольцами
        if np.isscalar(delta):
            delta_array = np.full(max(0, n - 1), delta)
        else:
            delta_array = np.asarray(delta, dtype=float)
            if delta_array.size != max(0, n - 1):
                raise ValueError(f"Размер массива delta ({delta_array.size}) должен быть равен n-1 ({n-1})")

        # Ёмкости на кольцах
        if np.isscalar(C):
            C_array = np.full(n * m, C)
        else:
            C_array = np.asarray(C, dtype=float)
            if C_array.size not in (n, n * m):
                raise ValueError(f"Размер массива C ({C_array.size}) должен быть n ({n}) или n*m ({n*m})")

        # Для дальнейших расчетов C в матрице используется по кольцу.
        if C_array.size == n:
            C_array = np.tile(C_array, m)

        # Построение матрицы индуктивностей
        L = inductance_matrix(n=n, m=m, R=R_array, L_own=L_own, A=A, delta=delta_array)

        Z_self = Z_self_matrix(r=r_ohm, C=C_array, n=n, m=m, R=R_array, omega=omega)
        U = generate_voltage_array(U_0, m, n, phi)
        I = calc_I(Z_self, U, omega, L, n, m)

        all_coords = points_on_rings_general(delta_array, n, A, N, R_array, m)

        # Формирование массива точек наблюдения
        grid_arr = np.asarray(grid)
        if grid_arr.ndim == 1:
            xx, yy = np.meshgrid(grid_arr, grid_arr, indexing='ij')
            obs_points = np.stack([xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())], axis=1)
        elif grid_arr.ndim == 2 and grid_arr.shape[1] == 2:
            obs_points = np.hstack([grid_arr, np.zeros((grid_arr.shape[0], 1))])
        elif grid_arr.ndim == 2 and grid_arr.shape[1] == 3:
            obs_points = grid_arr
        else:
            raise ValueError('grid должен быть 1D, 2D (x,y) или 2D (x,y,z) массивом')

        B = b_s_l(obs_points, I, N, n, m, all_coords)

        B_magnitude = np.linalg.norm(B, axis=1)
        std_B = np.std(B_magnitude)
        mean_B = np.mean(B_magnitude)
        B_target = self.fixed_params.get('B_target', 1e-5)

        # Метрика: неоднородность, делённая на "полезность" поля
        # Если mean_B << B_target → стоимость растёт
        # Если std_B велико → стоимость растёт
        quality_metric = (std_B + 1e-20) / (mean_B + 1e-20) * (B_target / (mean_B + 1e-20))

        return quality_metric

    def objective_function(self, positions: np.ndarray) -> np.ndarray:
        """Целевая функция для pyswarms.

        positions: массив размером (n_particles, n_dimensions). 
        Каждая частица представлена как [C, R, n, m, delta, phi].
        Возвращает: массив значений для каждой частицы.
        """
        n_particles = positions.shape[0]
        costs = np.zeros(n_particles)

        for i in range(n_particles):
            C = positions[i, 0]
            omega = positions[i, 1]
            phi = positions[i, 2]

            # Гарантия положительных значений
            if C <= 0 or omega <= 0:
                costs[i] = np.inf
                continue

            try:
                costs[i] = self.uniformity(C, omega, phi)
            except Exception:
                costs[i] = np.inf

        return costs

    def optimize(self, bounds, pso_options, n_particles: int = 30, max_iterations: int = 50):
        """
        Запуск оптимизации методом роя частиц

        Args:
            bounds: границы параметров [(min1, max1), (min2, max2), ...]
            n_particles: количество частиц
            max_iterations: максимальное количество итераций
            pso_options: параметры PSO

        Returns:
            Лучшая позиция и значение целевой функции
        """

        if pso_options is None:
            pso_options = {
                'c1': 0.5,
                'c2': 0.3,
                'w': 0.9,
            }

        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=len(bounds[0]),  # Теперь должно быть 3
            options=pso_options,
            bounds=bounds
        )

        best_cost, best_pos = optimizer.optimize(
            self.objective_function,
            iters=max_iterations,
            verbose=True
        )

        print(f"Лучшее значение: {best_cost}")
        print(f"Лучшие параметры:")
        print(f'\n C = {best_pos[0]:.2e} Ф')
        print(f'\n omega = {best_pos[1]:.2e} рад/с (f = {best_pos[1]/(2*np.pi):.2f} МГц)')
        print(f'\n phi = {best_pos[2]:.4f} рад')

        return best_pos, best_cost
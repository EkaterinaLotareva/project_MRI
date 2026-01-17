import numpy as np
from magnetic_field import *
from currents import *
from inductance import *
import pyswarms as ps


class MRIOptimizer:

    def __init__(self, fixed_params):
        self.fixed_params = fixed_params

    def uniformity(self, C, R, n, m, delta, phi):
        "Рассчитывает меру отклонения от однородности поля (чем меньше значение, тем более однородное -> ищем min)"
        r = self.fixed_params['r']
        rho = self.fixed_params['rho']
        A = self.fixed_params['A']
        U_0 = self.fixed_params['U_0']
        w = self.fixed_params['w']
        N = self.fixed_params['N']
        grid = self.fixed_params['grid']

        n = int(n)
        m = int(m)

        # Если R_params - скаляр, создаем массив одинаковых радиусов
        if np.isscalar(R):
            R_array = np.full(n * m, R)
        else:
            # Если уже массив, проверяем размер
            R_array = np.array(R)
            if len(R_array) != n * m:
                raise ValueError(f"Размер массива R ({len(R_array)}) должен быть равен n*m ({n * m})")


        delta_array = np.full(n-1, delta)
        C_array = np.full(n * m, C)


        L = inductance_matrix(n, m, R_array, r, A, delta_array)
        Z_self = Z_self_matrix(rho, r, C_array, n, m, R_array)
        U = generate_voltage_array(U_0, m, n, phi)
        I = calc_I(Z_self, U, w, L, n, m)
        B = b_s_l(grid, I, N, n, m, points_on_rings_general(delta_array, n, A, N, R_array, m))
        uniformity = np.std(B) / np.abs(np.mean(B))
        return uniformity

    def objective_function(self, positions: np.ndarray) -> np.ndarray:
        """
        Целевая функция для pyswarms
        positions: массив размером (n_particles, n_dimensions)
        Возвращает: массив значений для каждой частицы
        """
        n_particles = positions.shape[0]
        costs = np.zeros(n_particles)

        for i in range(n_particles):
            # Извлекаем параметры из позиции частицы
            # позиция = [C, R, n, m, delta, phi]
            C = positions[i, 0]
            R = positions[i, 1]
            n = positions[i, 2]
            m = positions[i, 3]
            delta = positions[i, 4]
            phi = positions[i, 5]

            n = int(n)
            m = int(m)

            if np.isscalar(R):
                R_array = np.full(n * m, R)
            else:
                # Если уже массив, проверяем размер
                R_array = np.array(R)
                if len(R_array) != n * m:
                    raise ValueError(f"Размер массива R ({len(R_array)}) должен быть равен n*m ({n * m})")

            delta_array = np.full(n - 1, delta)
            C_array = np.full(n * m, C)

            # Расчет неоднородности
            costs[i] = self.uniformity(C_array, R_array, n, m, delta_array, phi)

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

        # Параметры PSO по умолчанию
        if pso_options is None:
            pso_options = {
                'c1': 0.5,  # когнитивный параметр
                'c2': 0.3,  # социальный параметр
                'w': 0.9,  # инерционный вес
            }

        # Создание оптимизатора
        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=len(bounds[0]),
            options=pso_options,
            bounds=bounds
        )

        # Основной вызов оптимизации
        best_cost, best_pos = optimizer.optimize(
            self.objective_function,
            iters=max_iterations,
            verbose=True
        )

        # Результаты
        print(f"Лучшее значение: {best_cost}")
        print(f"Лучшие параметры:")
        print('\n C=' + str(best_pos[0]))
        print('\n R=' + str(best_pos[1]))
        print('\n n=' + str(best_pos[2]))
        print('\n m=' + str(best_pos[3]))
        print('\n delta=' + str(best_pos[4]))
        print('\n phi=' + str(best_pos[5]))

        return best_pos, best_cost

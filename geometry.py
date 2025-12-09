import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import *
from scipy import integrate
from scipy import constants


mu0=4*np.pi*1e-7
mu0_over_4pi = mu0 / (4 * np.pi) # Для удобства в расчетах

class SymmetricSystem:
    def __init__(self, m, R, delta, I_matrix, n, N=512, x_offset=0.0): # <-- Добавлен x_offset
        """
        Настройка системы.
        m - количество стопок.
        R - список радиусов колец (длиной n).
        delta - список зазоров между кольцами (длиной n-1).
        n - количество колец в стопке.
        N - количество точек на кольцо.
        x_offset - координата X центра первого кольца в стопке.
        """
        self.m = m
        self.angle_step = (2 * np.pi) / m  # Угол между стопками

        # 1. Проверки длин массивов
        if len(R) != n:
             raise ValueError(f"Длина R ({len(R)}) должна совпадать с количеством колец n ({n})")
        if n > 1 and len(delta) != n - 1:
             raise ValueError(f"Для n={n} колец список delta должен содержать {n-1} зазор(а). Длина delta: {len(delta)}")

        self.R = R
        self.delta = delta
        self.N = N
        self.I_matrix = np.asarray(I_matrix)
        self.n = n
        self.x_offset = x_offset

    def _get_centers(self):
        delta_array = np.array(self.delta)

        if self.n > 0:
            x_shifts = np.insert(delta_array, 0, self.x_offset)
        else:
            return np.empty((0, 3)) # Нет колец

        x_centers = np.cumsum(x_shifts)



        all_points = []
        theta = np.linspace(0, 2 * np.pi, self.N, endpoint=False)

        for i in range(self.n): # Итерируемся по n кольцам
            r = self.R[i]
            x_center = x_centers[i]

            # Кольцо в плоскости YZ
            y = r * np.cos(theta)
            z = r * np.sin(theta)

            # X-координата центра кольца
            x_coords = np.full_like(y, x_center)

            all_points.append(np.vstack([x_coords, y, z]).T)

        return np.vstack(all_points)

    def _rotate_points(self, coords, phi):
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        Rz = np.array([
            [cos_phi, -sin_phi, 0],
            [sin_phi,  cos_phi, 0],
            [0,         0,       1]
        ])
        return coords @ Rz


    def get_ring_center(self, stack_index, ring_index):

        delta_array = np.array(self.delta)
        x_centers_all = np.cumsum(np.insert(delta_array, 0, self.x_offset))

        if ring_index >= self.n:
             raise IndexError("Индекс кольца выходит за пределы")

        x_j = x_centers_all[ring_index]
        C_local = np.array([x_j, 0, 0])
        angle_phi = stack_index * self.angle_step
        return self._rotate_points(C_local, angle_phi)

    def assemble(self):
        system_coords = []
        base_part = self._get_centers()

        for i in range(self.m):
            current_angle = i * self.angle_step
            rotated_part = self._rotate_points(base_part, current_angle)
            system_coords.append(rotated_part)

        return np.vstack(system_coords)

    def b_s_l(self, obs_points):
        if self.I_matrix is None:
            raise ValueError("Матрица токов не задана")

        obs_points = np.array(obs_points)
        if obs_points.ndim == 1:
            obs_points = obs_points[np.newaxis, :]


        B = np.zeros((obs_points.shape[0], 3))

        all_coordinates = self.assemble()

        for i_stack in range(self.m):
            for i_ring in range(self.n):
                I = self.I_matrix[i_stack, i_ring]
                if I == 0: continue # Пропускаем кольца с нулевым током

                # Индексация
                start_index = (i_stack * self.n * self.N) + (i_ring * self.N)
                end_index = start_index + self.N
                ring_coords = all_coordinates[start_index:end_index]

                # Сегментация и замыкание
                P1 = ring_coords[:-1, :]
                P2 = ring_coords[1:, :]
                P1 = np.vstack([P1, ring_coords[-1, :]])
                P2 = np.vstack([P2, ring_coords[0, :]])

                # Расчет dl и P_mid
                dl_vector = P2 - P1
                P_mid = (P1 + P2) / 2
                P_mid_reshaped = P_mid[:, np.newaxis, :]

                # Расчет r
                r_vector = obs_points - P_mid_reshaped
                r_mag_cubed = np.linalg.norm(r_vector, axis=2, keepdims=True)**3
                r_mag_cubed = np.where(r_mag_cubed < 1e-12, 1e-12, r_mag_cubed) # Защита от деления на ноль

                # Закон Био-Савара
                dl_cross_r = np.cross( dl_vector[:, np.newaxis, :], r_vector)

                dB_segments = mu0_over_4pi * I * dl_cross_r / r_mag_cubed

                # Накопление
                B += np.sum(dB_segments, axis=0)

        return B

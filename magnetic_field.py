import numpy as np
import geometry

mu0=4*np.pi*1e-7
mu0_over_4pi = mu0 / (4 * np.pi) # Для удобства в расчетах

def b_s_l(obs_points, I_matrix, N, n, m, all_coordinates):

    if I_matrix is None:
        raise ValueError("Матрица токов не задана")

    obs_points = np.array(obs_points)
    if obs_points.ndim == 1:
        obs_points = obs_points[np.newaxis, :]

    B = np.zeros((obs_points.shape[0], 3))

    for i_stack in range(m):
        for i_ring in range(n):
            I = I_matrix[i_stack, i_ring]
            if I == 0: continue  # Пропускаем кольца с нулевым током

            # Индексация
            start_index = (i_stack * n * N) + (i_ring * N)
            end_index = start_index + N
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
            r_mag_cubed = np.linalg.norm(r_vector, axis=2, keepdims=True) ** 3
            r_mag_cubed = np.where(r_mag_cubed < 1e-12, 1e-12, r_mag_cubed)  # Защита от деления на ноль

            # Закон Био-Савара
            dl_cross_r = np.cross(dl_vector[:, np.newaxis, :], r_vector)

            dB_segments = mu0_over_4pi * I * dl_cross_r / r_mag_cubed

            # Накопление
            B += np.sum(dB_segments, axis=0)

    return B
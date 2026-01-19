import numpy as np
import geometry

mu0=4*np.pi*1e-7
mu0_over_4pi = mu0 / (4 * np.pi) # Для удобства в расчетах


def points_on_rings_one_stack(delta, n, A, N, R):
    delta_array = np.array(delta)

    if n > 0:
        x_shifts = np.insert(delta_array, 0, A)
    else:
        return np.empty((0, 3))  # Нет колец

    x_centers = np.cumsum(x_shifts)
    all_points = []
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

    for i in range(n):  # Итерируемся по n кольцам
        r = R[i]
        x_center = x_centers[i]

        # Кольцо в плоскости YZ
        y = r * np.cos(theta)
        z = r * np.sin(theta)

        # X-координата центра кольца
        x_coords = np.full_like(y, x_center)

        all_points.append(np.vstack([x_coords, y, z]).T)

    return np.vstack(all_points)


def rotate_points( coords, phi):
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    Rz = np.array([
        [cos_phi, -sin_phi, 0],
        [sin_phi, cos_phi, 0],
        [0, 0, 1]
    ])
    return coords @ Rz


def ring_center_general(delta, A, n, fi, stack_index, ring_index):
    delta_array = np.array(delta)
    x_centers_all = np.cumsum(np.insert(delta_array, 0, A))

    if ring_index >= n:
        raise IndexError("Индекс кольца выходит за пределы")

    x_j = x_centers_all[ring_index]
    C_local = np.array([x_j, 0, 0])
    angle_phi = stack_index * fi
    return rotate_points(C_local, angle_phi)


def points_on_rings_general(delta, n, A, N, R, m):
    fi = (2 * np.pi) / m
    system_coords = []
    base_part = points_on_rings_one_stack(delta, n, A, N, R)

    for i in range(m):
        current_angle = i * fi
        rotated_part = rotate_points(base_part, current_angle)
        system_coords.append(rotated_part)

    return np.vstack(system_coords)

def b_s_l(obs_points, I_matrix, N, n, m, all_coordinates):

    if I_matrix is None:
        raise ValueError("Матрица токов не задана")

    obs_points = np.array(obs_points)

    #Если передана одна точка [x, y] или [x, y, z], делаем из неё строку (1, N)
    if obs_points.ndim == 1:
        obs_points = obs_points[np.newaxis, :]

    # Если координат 2 (x, y), добавляем координаты по Z со значениями 0
    if obs_points.shape[1] == 2:
        z_zeros = np.zeros((obs_points.shape[0], 1))
        obs_points = np.hstack([obs_points, z_zeros])

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
            B += np.real(np.sum(dB_segments, axis=0))

    return B

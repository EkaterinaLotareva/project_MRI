import numpy as np

def points_on_rings_one_stack(delta, n, A, N, R):
    delta_array = np.array(delta)
    if n > 0:
        x_shifts = np.insert(delta_array, 0, A)
    else:
        return np.empty((0, 3))

    x_centers = np.cumsum(x_shifts)
    all_points = []
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

    for i in range(n):
        r = R[i]
        x_center = x_centers[i]
        y = r * np.cos(theta)
        z = r * np.sin(theta)
        x_coords = np.full_like(y, x_center)
        all_points.append(np.vstack([x_coords, y, z]).T)

    return np.vstack(all_points)

def rotate_points(coords, phi):
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    Rz = np.array([
        [cos_phi, sin_phi, 0],
        [-sin_phi, cos_phi, 0],
        [0, 0, 1]
    ])
    return coords @ Rz

def ring_center_general(delta, A, n, fi, stack_index, ring_index):
    delta_array = np.array(delta)
    x_centers_all = np.cumsum(np.insert(delta_array, 0, A))
    if ring_index >= n:
        raise IndexError("Индекс кольца выходит за пределы")
    x_j = x_centers_all[ring_index]
    C_local = np.array([x_j, 0.0, 0.0])
    angle_phi = stack_index * fi
    return rotate_points(C_local, angle_phi)

def all_rings_centers(delta, A, n, m, fi):
    """Возвращает координаты центров всех n*m колец (массив формы (m*n, 3))"""
    centers = np.zeros((m * n, 3))
    for s_idx in range(m):
        for r_idx in range(n):
            centers[s_idx * n + r_idx] = ring_center_general(delta, A, n, fi, s_idx, r_idx)
    return centers

def points_on_rings_general(delta, n, A, N, R, m):
    fi = (2 * np.pi) / m
    system_coords = []
    base_part = points_on_rings_one_stack(delta, n, A, N, R)
    system_normals = []
    base_normal = np.array([-1.0, 0.0, 0.0])
    for i in range(m):
        current_angle = i * fi
        rotated_part = rotate_points(base_part, current_angle)
        rotated_normal = rotate_points(base_normal, current_angle)
        system_coords.append(rotated_part)
        system_normals.append(rotated_normal)

    return np.vstack(system_coords), np.vstack(system_normals)
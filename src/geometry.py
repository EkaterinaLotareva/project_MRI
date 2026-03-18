import numpy as np

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


def get_ring_normal(points):
    center = np.mean(points, axis=0)
    r_vec = points[0] - center
    dl_vec = points[1] - points[0]
    n = np.cross(r_vec, dl_vec)
    return n / np.linalg.norm(n)
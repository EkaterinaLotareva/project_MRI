import numpy as np

class SymmetricSystem:
    def __init__(self, m, R, delta, n, A, N=512):
        """
        Настройка системы:
        m - количество стопок
        n - количество колец в стопке
        R - список радиусов колец (длиной n) (от центра к краю)
        delta - список зазоров между кольцами (длиной n-1) (от центра к краю)
        A - расстояние от центра до первого кольца в стопке
        N - количество точек на кольцо
        """
        self.m = m
        self.fi = (2 * np.pi) / m  # Угол между стопками
        self.R = R
        self.delta = delta
        self.N = N
        self.n = n
        self.A = A

        '''Проверка длин массивов'''
        if len(R) != n:
             raise ValueError(f"Длина R ({len(R)}) должна совпадать с количеством колец n ({n})")
        if n > 1 and len(delta) != n - 1:
             raise ValueError(f"Для n={n} колец список delta должен содержать {n-1} зазор(а). Длина delta: {len(delta)}")

    def points_on_rings_one_stack(self):

        delta_array = np.array(self.delta)

        if self.n > 0:
            x_shifts = np.insert(delta_array, 0, self.A)
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


    def rotate_points(self, coords, phi):
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        Rz = np.array([
            [cos_phi, -sin_phi, 0],
            [sin_phi,  cos_phi, 0],
            [0,         0,       1]
        ])
        return coords @ Rz


    def ring_center_general(self, stack_index, ring_index):

        delta_array = np.array(self.delta)
        x_centers_all = np.cumsum(np.insert(delta_array, 0, self.A))

        if ring_index >= self.n:
             raise IndexError("Индекс кольца выходит за пределы")

        x_j = x_centers_all[ring_index]
        C_local = np.array([x_j, 0, 0])
        angle_phi = stack_index * self.fi
        return self.rotate_points(C_local, angle_phi)

    def points_on_rings_general(self):
        system_coords = []
        base_part = self.points_on_rings_one_stack()

        for i in range(self.m):
            current_angle = i * self.fi
            rotated_part = self.rotate_points(base_part, current_angle)
            system_coords.append(rotated_part)

        return np.vstack(system_coords)





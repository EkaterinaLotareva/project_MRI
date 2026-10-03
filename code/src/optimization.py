import numpy as np
import pyswarms as ps

from src.geometry import points_on_rings_general, all_rings_centers
from src.magnetic_field import quality_metric
from src.currents import Z_self_matrix, generate_voltage_array, calc_I


from src.inductance_quasystat import inductance_matrix_quasystat
from src.inductance import inductance_matrix_nonqs

C_LIGHT = 299792458.0


class MRIOptimizer:
    def __init__(self, fixed_params):
        self.fp = fixed_params
        self.cost_history = []
        self.best_pos = None
        self.best_cost = None

        mode = self.fp.get('calculation_mode', 'quasistatic').lower()

    def unpack_position(self, pos):
        n = int(self.fp['n'])
        C_scalar = pos[0]
        radii = np.array(pos[1 : 1 + n])
        deltas = np.array(pos[1 + n : 1 + n + (n - 1)]) if n > 1 else np.array([])
        return C_scalar, radii, deltas

    def evaluate_particle(self, pos):
        n = int(self.fp['n'])
        m = int(self.fp['m'])
        A = float(self.fp['A'])
        omega = float(self.fp['omega'])
        r_ohm = float(self.fp['r_ohm'])
        U_0 = float(self.fp['U_0'])
        L_own = float(self.fp['L_own'])
        phi = float(self.fp.get('phi', 0.0))
        R_domain = float(self.fp.get('R_domain', 0.5 * A))

        C_scalar, radii, deltas = self.unpack_position(pos)

        if np.any(radii <= 1e-4) or (len(deltas) > 0 and np.any(deltas <= 1e-5)) or C_scalar <= 1e-13:
            return 1e6

        try:
            fi = 2 * np.pi / m
            all_coords, normals = points_on_rings_general(
                delta=deltas, n=n, A=A, N=self.fp.get('N', 128), R=radii, m=m
            )
            centers = all_rings_centers(delta=deltas, A=A, n=n, m=m, fi=fi)

            L = self.inductance_func(
                n=n, m=m, R=radii, L_own=L_own, A=A, delta=deltas
            )

            # Расчет токов
            C_array = np.full(n * m, C_scalar)
            Z_self = Z_self_matrix(r=r_ohm, C=C_array, n=n, m=m, R=radii, omega=omega)
            U = generate_voltage_array(U_0=U_0, m=m, n=n, phi_0=phi)
            I_matrix = calc_I(Z_self=Z_self, U=U, omega=omega, L=L, n=n, m=m)

            if I_matrix is None or not np.all(np.isfinite(I_matrix)):
                return 1e6

            # Оценка однородности CV
            cv_cost = quality_metric(
                I=I_matrix, n=n, m=m,
                current_centers=centers, current_normals=normals,
                R_array=radii, R_domain=R_domain, N_grid=20
            )
            return float(cv_cost)

        except Exception:
            return 1e6

    def objective_function(self, positions):
        costs = np.zeros(positions.shape[0])
        for i in range(positions.shape[0]):
            costs[i] = self.evaluate_particle(positions[i])
        return costs

    def optimize(self, bounds, pso_options, n_particles=30, max_iterations=40):
        dimensions = len(bounds[0])
        self.optimizer_pso = ps.single.GlobalBestPSO(
            n_particles=n_particles, dimensions=dimensions,
            options=pso_options, bounds=bounds
        )
        self.best_cost, self.best_pos = self.optimizer_pso.optimize(
            self.objective_function, iters=max_iterations, verbose=True
        )
        self.cost_history = self.optimizer_pso.cost_history
        return self.best_pos, self.best_cost
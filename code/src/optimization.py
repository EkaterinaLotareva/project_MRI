import logging
import numpy as np
import pyswarms as ps

from src.geometry import points_on_rings_general, all_rings_centers, stack_basis
from src.magnetic_field import quality_metric_std, b_s_l_optimized
from src.currents import Z_self_matrix, generate_voltage_array, calc_I
from src.resonance_find import find_resonance_frequency
from src.inductance_quasystat import inductance_matrix_quasystat, set_geometry
from src.nonqs_inductance import inductance_matrix_nonqs

logger = logging.getLogger(__name__)


class MRIOptimizer:
    def __init__(self, fixed_params):
        self.fp = fixed_params
        self.cost_history = []
        self.best_pos = None
        self.best_cost = None

        mode = self.fp.get('calculation_mode', 'quasistatic').lower()
        if mode not in {'quasistatic', 'nonquasistatic'}:
            raise ValueError(
                "calculation_mode должен быть 'quasistatic' или 'nonquasistatic'"
            )
        self.calculation_mode = mode

    def unpack_position(self, pos):
        n = int(self.fp['n'])
        radii = np.array(pos[0 : n])
        deltas = np.array(pos[n : n + (n - 1)]) if n > 1 else np.array([])
        return radii, deltas

    def evaluate_particle(self, pos):
        n = int(self.fp['n'])
        m = int(self.fp['m'])
        A = float(self.fp['A'])
        C_array = np.asarray(self.fp['C'])
        omega = float(self.fp['omega'])
        r_ohm = float(self.fp['r_ohm'])
        U_0 = float(self.fp['U_0'])
        L_own = float(self.fp['L_own'])
        phi = float(self.fp.get('phi', 0.0))
        R_domain = float(self.fp.get('R_domain', 0.5 * A))

        f_min = float(self.fp.get('f_min', 250e6))
        f_max = float(self.fp.get('f_max', 350e6))
        B_target = float(self.fp.get('B_target', 1e-6))

        radii, deltas = self.unpack_position(pos)

        if (not np.all(np.isfinite(pos)) or np.any(radii <= 1e-4)
                or (len(deltas) > 0 and np.any(deltas <= 1e-5))):
            return 1e6

        if np.any(C_array <= 0) or np.any(C_array > 1e-7):
            return 1e6

        for i_s in range(m):
            for j_s in range(i_s + 1, m):
                d = 2 * A * np.sin(np.pi * (j_s - i_s) / m)
                for i_r in range(n):
                    for j_r in range(n):
                        if radii[i_r] + radii[j_r] > d * 0.95:
                            return 1e6

        try:
            fi = 2 * np.pi / m
            _, normals = points_on_rings_general(
                delta=deltas, n=n, A=A, N=128, R=radii, m=m
            )
            centers = all_rings_centers(delta=deltas, A=A, n=n, m=m, fi=fi)
            e1, e2 = stack_basis(m)

            if self.calculation_mode == 'quasistatic':
                set_geometry(m=m, n=n, ring_centers=centers,
                             e1=e1, e2=e2, radii=radii, L_own=L_own)
                L = inductance_matrix_quasystat(f_hz=omega / (2 * np.pi))
            else:
                L = inductance_matrix_nonqs(
                    ring_centers=centers, normals=normals, e1=e1, e2=e2,
                    radii=radii, n=n, m=m, L_own=L_own, f=omega / (2 * np.pi),
                )

            f_res, f_fine, B_fine, peaks_freqs = f_res, f_fine, B_fine, peaks_freqs = find_resonance_frequency(
                L=L, n=n, m=m, radii=radii,
                A=A, gaps=deltas,
                C_value=C_array, r_ohm=r_ohm, U_0=U_0,
                ring_centers=centers, normals=normals,
                obs_point=np.array([[0.0, 0.0, 0.0]]),
                f_min=f_min, f_max=f_max,
                n_coarse=10, n_fine=4000,
                aaa_tol=1e-8, aaa_mmax=80,
                prominence_frac=0.01,
                return_curve=True, return_all_peaks=True,
            )
            if not (f_min < f_res < f_max):
                return 1e6

            mag_fine = np.linalg.norm(B_fine, axis=1)
            mag_at_res = mag_fine[np.argmin(np.abs(f_fine - f_res))]
            mag_max = mag_fine.max()

            if mag_max <= 0 or mag_at_res < 0.3 * mag_max:
                return 1e6

            omega_res = f_res * 2 * np.pi
            Z_self = Z_self_matrix(r=r_ohm, C=C_array, n=n, m=m,
                                   R=radii, omega=omega_res)
            U = generate_voltage_array(U_0=U_0, m=m, n=n, phi_0=phi)
            I_matrix = calc_I(Z_self=Z_self, U=U, omega=omega_res, L=L, n=n, m=m)

            if I_matrix is None or not np.all(np.isfinite(I_matrix)):
                return 1e6

            cv_cost = quality_metric_std(
                I=I_matrix, n=n, m=m,
                current_centers=centers, current_normals=normals,
                R_array=radii, R_domain=R_domain, N_grid=15,
            )

            if not np.isfinite(cv_cost):
                return 1e6

            B_center = b_s_l_optimized(
                np.array([[0.0, 0.0, 0.0]]), I_matrix,
                n, m, centers, normals, radii
            )
            B_mag = np.abs(B_center[0, 0])
            field_penalty = max(1.0, (B_target / (B_mag + 1e-12)) ** 2)

            return float(cv_cost * field_penalty)

        except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
            logger.warning("Ошибка частицы: %s", exc)
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
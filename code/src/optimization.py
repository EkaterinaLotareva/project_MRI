import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from src.geometry import points_on_rings_general, ring_center_general
from src.magnetic_field import b_s_l_optimized, quality_metric
from src.currents import Z_self_matrix, generate_voltage_array, calc_I
from src.inductance import inductance_matrix
import warnings
from pyswarms.utils.plotters import plot_cost_history

class MRIOptimizer:
    def __init__(self, fixed_params):
        self.fp = fixed_params
        self.L = None 
        self.ring_centers = None
        self.all_coordinates = None
        self.normals = None

    def uniformity(self, C_scalar, omega, current_L, current_centers, current_normals):
        fp = self.fp
        r_ohm, A, U_0, R = fp['r_ohm'], fp['A'], fp['U_0'], fp['R']
        n, m = int(fp['n']), int(fp['m'])
        
        # 1. Расчет токов
        R_array = np.full(n, R) if np.isscalar(R) else np.asarray(R)
        C_array = np.full(n * m, C_scalar)
        Z_self = Z_self_matrix(r=r_ohm, C=C_array, n=n, m=m, R=R_array, omega=omega)
        U = generate_voltage_array(U_0, m, n, fp['phi'])
        I = calc_I(Z_self, U, omega, current_L, n, m)
        
        if I is None or not np.all(np.isfinite(I)) or np.all(np.abs(I) < 1e-12):
            return 1e10
        
        R_domain = fp['R_domain'] if 'R_domain' in fp else 0.01  
    
        cost = quality_metric(
            I=I, 
            n=n, 
            m=m, 
            current_centers=current_centers, 
            current_normals=current_normals, 
            R_array=R_array, 
            R_domain=R_domain
        )
    
        return cost
       

    def objective_function(self, positions):
        positions = np.atleast_2d(positions)
        costs = np.zeros(positions.shape[0])
            
        n = int(self.fp['n'])
        m = int(self.fp['m'])
        A = self.fp['A']


        for i in range(positions.shape[0]):
        

            C = positions[i, 0]
            d1 = positions[i, 1]
            d2 = positions[i, 2]
            d3 = positions[i, 3]
            d4 = positions[i, 4]
            
            deltas = np.array([d1, d2, d3, d4])
            all_coords, normals = points_on_rings_general(
                    delta=deltas, n=n, A=A, N=self.fp['N'], R=self.fp['R'], m=m
                )
                    
            L_new = inductance_matrix(
                    n=n, m=m, R=self.fp['R'], L_own=self.fp['L_own'], 
                    A=A, delta=deltas, all_points=all_coords, normals=normals, N_seg=self.fp['N']
                )
            

            fi = (2 * np.pi) / m
            centers = np.zeros((m * n, 3))
            for s_idx in range(m):
                for r_idx in range(n):
                    centers[s_idx * n + r_idx] = ring_center_general(deltas, A, n, fi, s_idx, r_idx)
            
            cost = self.uniformity(C, self.fp['omega'], L_new, centers, normals)
            
            costs[i] = cost 
            
        return costs 
    
    def optimize(self, bounds, pso_options, n_particles=500, max_iterations=500):
        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=len(bounds[0]),
            options=pso_options,
            bounds=bounds
        )
        best_cost, best_pos = optimizer.optimize(self.objective_function, iters=max_iterations)
     


        plot_cost_history(cost_history=optimizer.cost_history)
        plt.title("Оптимизация однородности поля")
        plt.xlabel("Итерации")
        plt.ylabel("Значение целевой функции")
        plt.show()
        
        return best_pos, best_cost
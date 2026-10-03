from src.visualization import plot_optimal_configuration
from src.geometry import points_on_rings_general, all_rings_centers
import os
import json
import numpy as np
import matplotlib.pyplot as plt

import src.config as config
from src.optimization import MRIOptimizer
from src.geometry import all_rings_centers
from src.utils import save_optimization_session
from src.visualization import plot_field_along_axis
from src.visualization import visualize_rings



N_PARTICLES = 30           # Число частиц роя 
MAX_ITERATIONS = 45        # Число итераций 
PSO_OPTIONS = {
    'c1': 1.7,             # Когнитивный вес (тяга к личному рекорду)
    'c2': 1.7,             # Социальный вес (тяга к рекорду роя)
    'w': 0.7               # Инерция (размер «шага» и скорость движения)
}


CALCULATION_MODE = 'quasistatic'  


k = 0.50        


OVERRIDE_CONFIG = True
M_STACKS = 6
N_RINGS = 5
A_RATIO = 0.08


C_BOUNDS = (5e-12, 50e-12)       # Емкость: 5 - 50 пФ
R_BOUNDS = (0.020, 0.050)        # Радиусы колец: 20 - 50 мм
DELTA_BOUNDS = (0.001, 0.040)    # Зазоры между кольцами вдоль стопки: 1 - 40 мм




def run_pso_optimization():

    

    fp = config.get_fixed_params()
    
    if OVERRIDE_CONFIG:
        fp['m'] = int(M_STACKS)
        fp['n'] = int(N_RINGS)
        fp['A'] = A_RATIO
    
    m = int(fp['m'])
    n = int(fp['n'])
    A = float(fp['A'])
    fi = 2 * np.pi / m
    

    R_domain = k * A
    fp['R_domain'] = R_domain
    fp['calculation_mode'] = CALCULATION_MODE
    
    print(f"Режим расчета поля:     {CALCULATION_MODE.upper()}")
    print(f"Геометрия катушки:      стопок m = {m}, колец в стопке n = {n}")
    print(f"Базовый радиус A:       {A * 1e3:.1f} мм")
    print(f"Область оценки CV:      круг радиусом {R_domain * 1e3:.1f} мм ({k * 100:.0f}% от A)")
    print(f"Параметры роя (PSO):    {N_PARTICLES} частиц, {MAX_ITERATIONS} итераций")
    print("-" * 65)

    optimizer = MRIOptimizer(fp)
    
    lower_bounds = [C_BOUNDS[0]] + [R_BOUNDS[0]] * n
    upper_bounds = [C_BOUNDS[1]] + [R_BOUNDS[1]] * n
    
    if n > 1:
        lower_bounds += [DELTA_BOUNDS[0]] * (n - 1)
        upper_bounds += [DELTA_BOUNDS[1]] * (n - 1)
        
    bounds = (np.array(lower_bounds), np.array(upper_bounds))
    

    best_pos, best_cost = optimizer.optimize(
        bounds=bounds,
        pso_options=PSO_OPTIONS,
        n_particles=N_PARTICLES,
        max_iterations=MAX_ITERATIONS
    )
    

    opt_C, opt_radii, opt_deltas = optimizer.unpack_position(best_pos)
    
    if len(opt_deltas) > 0:
        stack_x_centers = np.cumsum(np.insert(opt_deltas, 0, A))
    else:
        stack_x_centers = np.array([A])
        
    global_centers = all_rings_centers(delta=opt_deltas, A=A, n=n, m=m, fi=fi)
    

    print(f"Итоговый CV (std/mean):        {best_cost:.4e} ({best_cost * 100:.2f} %)")
    print(f"Оптимальная емкость C:          {opt_C * 1e12:.2f} пФ")

    
    print("Радиусы колец вдоль стопки (R):")
    for i, r in enumerate(opt_radii):
        print(f"  Кольцо {i + 1}: R = {r * 1e3:6.2f} мм")
        
    if len(opt_deltas) > 0:
        print("Зазоры между соседними кольцами в стопке (deltas):")
        for i, d in enumerate(opt_deltas):
            print(f"  Между кольцами {i + 1} и {i + 2}: d = {d * 1e3:6.2f} мм")
            

    
    opt_folder = save_optimization_session(
        best_pos=best_pos,
        best_cost=best_cost,
        cost_history=optimizer.cost_history,
        fixed_params=fp,
        pso_options=PSO_OPTIONS
    )
    
    # Сохранение полной конфигурации в JSON
    optimized_config = {
        'calculation_mode': CALCULATION_MODE,
        'ROI_fraction': k,
        'R_domain_m': float(R_domain),
        'CV_metric': float(best_cost),
        'm': m,
        'n': n,
        'A': A,
        'C': float(opt_C),
        'radii': opt_radii.tolist(),
        'deltas': opt_deltas.tolist(),
        'stack_centers_radial': stack_x_centers.tolist(),
        'all_ring_centers_3d': global_centers.tolist()
    }
    
    json_path = os.path.join(opt_folder, "optimized_config.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(optimized_config, f, indent=4, ensure_ascii=False)
        
    # Построение графика сходимости
    plt.figure(figsize=(8, 5))
    plt.plot(optimizer.cost_history, color='tab:blue', lw=2)
    plt.title(f"Сходимость CV", fontsize=12)
    plt.xlabel("Итерация")
    plt.ylabel("CV")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(opt_folder, "convergence_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    run_pso_optimization()
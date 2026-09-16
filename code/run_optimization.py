# run_optimization.py
import os
import numpy as np
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_cost_history

import src.config as config
from src.optimization import MRIOptimizer
from src.utils import save_optimization_session


def run_pso_optimization():
    
    fp = config.get_fixed_params()
    n = int(fp['n'])
    m = int(fp['m'])
    A = fp['A']
    
    optimizer = MRIOptimizer(fp)
    
    C_min, C_max = 5e-12, 50e-12
    R_min, R_max = 0.020, 0.050
    delta_min, delta_max = 0.001, 0.040

    lower_bounds = [C_min] + [R_min] * n + [delta_min] * (n - 1)
    upper_bounds = [C_max] + [R_max] * n + [delta_max] * (n - 1)
    bounds = (np.array(lower_bounds), np.array(upper_bounds))
    
    options = {'c1': 1.8, 'c2': 1.8, 'w': 0.8}
    
    n_particles = 25
    max_iterations = 35
    
    total_dims = len(lower_bounds)
    
    best_pos, best_cost = optimizer.optimize(
        bounds=bounds,
        pso_options=options,
        n_particles=n_particles,
        max_iterations=max_iterations
    )
    
    opt_C, opt_radii, opt_deltas = optimizer.unpack_position(best_pos)
    
    ring_positions_x = np.cumsum(np.insert(opt_deltas, 0, A))
    
    print("\n" + "=" * 55)
    print("         РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ (PSO)")
    print("=" * 55)
    print(f"Целевая функция CV (std/mean): {best_cost:.4e} ({best_cost * 100:.2f} %)")
    print(f"Оптимальная емкость C:          {opt_C * 1e12:.2f} пФ")
    print("-" * 55)
    print("Радиусы колец (R):")
    for i, r in enumerate(opt_radii):
        print(f"  Кольцо {i + 1}: R = {r * 1e3:6.2f} мм  ({r:6.4f} м)")
    
    print("-" * 55)
    print("Зазоры между кольцами (deltas):")
    for i, d in enumerate(opt_deltas):
        print(f"  Зазор между кольцами {i + 1} и {i + 2}: d = {d * 1e3:6.2f} мм  ({d:6.4f} м)")
    
    print("-" * 55)
    print("Положение центров колец от центра системы (ось луча):")
    for i, pos_x in enumerate(ring_positions_x):
        print(f"  Кольцо {i + 1}: X_center = {pos_x * 1e3:6.2f} мм")
    print("=" * 55 + "\n")
    
    opt_folder = save_optimization_session(
        best_pos=best_pos,
        best_cost=best_cost,
        cost_history=optimizer.cost_history,
        fixed_params=fp,
        pso_options=options
    )
    
    summary_txt_path = os.path.join(opt_folder, "optimal_configuration.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(f"CV: {best_cost:.6e}\n")
        f.write(f"Емкость C: {opt_C * 1e12:.3f} pF\n\n")
        f.write("Радиусы колец:\n")
        for i, r in enumerate(opt_radii):
            f.write(f"  R_{i+1}: {r*1e3:.2f} mm\n")
        f.write("\nЗазоры между соседними кольцами:\n")
        for i, d in enumerate(opt_deltas):
            f.write(f"  delta_{i+1}_{i+2}: {d*1e3:.2f} mm\n")
        f.write("\nАбсолютные координаты центров (вдоль луча):\n")
        for i, px in enumerate(ring_positions_x):
            f.write(f"  X_center_{i+1}: {px*1e3:.2f} mm\n")
            
    print(f"Текстовый отчет сохранен в: {summary_txt_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(optimizer.cost_history, color='tab:red', lw=2)
    plt.title("Сходимость оптимизации коэф.вариации", fontsize=13)
    plt.xlabel("Номер терации", fontsize=11)
    plt.ylabel("Значение коэф.вариации", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = os.path.join(opt_folder, "convergence_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График сходимости сохранен в: {plot_path}")


if __name__ == '__main__':
    run_pso_optimization()
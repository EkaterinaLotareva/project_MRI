# run_optimization.py
import numpy as np
import matplotlib.pyplot as plt
from src.optimization import MRIOptimizer
from pyswarms.utils.plotters import plot_cost_history
import pyswarms as ps
import src.config as config  
import os
from src.utils import save_optimization_session


def run_pso_optimization():
    print("Инициализация оптимизатора роя частиц...")
    
    # Получаем базовые параметры
    fp = config.get_fixed_params()
    optimizer = MRIOptimizer(fp)
    
    # Настройка границ для параметров: [C, delta1, delta2, delta3, delta4]
    lower_bounds = [1e-11, 0.005, 0.0005, 0.005, 0.0005]
    upper_bounds = [5e-11, 0.1, 0.01, 0.1, 0.01]
    bounds = (np.array(lower_bounds), np.array(upper_bounds))
    
    # Параметры алгоритма PSO
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    
    print("Запуск итерационного процесса...")
    best_pos, best_cost = optimizer.optimize(
        bounds=bounds, 
        pso_options=options, 
        n_particles=500, 
        max_iterations=500
    )
    
    print("\n=== ОПТИМИЗАЦИЯ ЗАВЕРШЕНА ===")
    print(f"Лучшее значение целевой функции (Cost): {best_cost:.4e}")
    print(f"Оптимальная емкость C: {best_pos[0]:.2e} Ф")
    print(f"Оптимальные зазоры deltas: {best_pos[1:]} м")
    
    # Сохранение графика сходимости целевой функции
    plot_cost_history(cost_history=optimizer.optimize.cost_history)
    plt.title("История сходимости алгоритма PSO") 
    plt.savefig("optimization_cost_history.png", dpi=300)
    plt.close()
    print("График истории сходимости сохранен в 'optimization_cost_history.png'")

    optimizer_instance = MRIOptimizer(fp)
    
    
    print("Запуск оптимизации роем частиц...")
    best_cost, best_pos = optimizer.optimize(
        bounds=bounds, 
        options=options, 
        n_particles=30, 
        max_iterations=30
    )
    
    cost_history = optimizer.optimizer.cost_history 
    
    # Вызываем нашу новую функцию сохранения
    opt_folder = save_optimization_session(
        best_pos=best_pos,
        best_cost=best_cost,
        cost_history=cost_history,
        fixed_params=fp,
        pso_options=options
    )
    
    # сразу строим и сохраняем график сходимости в эту же папку
    from pyswarms.utils.plotters import plot_cost_history
    import matplotlib.pyplot as plt
    
    plot_cost_history(cost_history=cost_history)
    plt.title("Convergence Curve (PSO)")
    plt.savefig(os.path.join(opt_folder, "convergence_plot.png"), dpi=300)
    plt.close()

if __name__ == '__main__':
    run_pso_optimization()
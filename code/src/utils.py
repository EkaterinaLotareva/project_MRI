import os
import json
from datetime import datetime
import numpy as np

def create_run_directory(base_results_dir="./results"):
    
    #Создание папки для сохранения результатов текущего запуска с уникальным именем на основе временной метки.
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_results_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_simulation_data(run_dir, config_dict, matrices_dict):
    """
    Сохраняет конфигурацию эксперимента и тяжелые матрицы данных.
    
    :param run_dir: Путь к папке текущего запуска
    :param config_dict: Словарь с физическими параметрами 
    :param matrices_dict: Словарь с массивами NumPy 
    """
    serializable_config = {}
    for k, v in config_dict.items():
        if isinstance(v, np.ndarray):
            serializable_config[k] = v.tolist()
        elif hasattr(v, 'item'):
            serializable_config[k] = v.item()
        else:
            serializable_config[k] = v

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(serializable_config, f, indent=4, ensure_ascii=False)
    
    data_path = os.path.join(run_dir, "simulation_data.npz")
    np.savez_compressed(data_path, **matrices_dict)
    
    print(f"\n[Data Logging] Результаты успешно зафиксированы в: {run_dir}")

def save_optimization_session(best_pos, best_cost, cost_history, fixed_params, pso_options, base_dir="./optimization_results"):
    """
    Сохраняет все метаданные, результаты и историю сходимости PSO-оптимизации.
    """
    # Создаем уникальную папку для этой сессии оптимизации
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"opt_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    
    sanitized_fp = {}
    for k, v in fixed_params.items():
        if isinstance(v, np.ndarray):
            sanitized_fp[k] = v.tolist()
        elif hasattr(v, 'item'):
            sanitized_fp[k] = v.item()
        else:
            sanitized_fp[k] = v

    summary_data = {
        "timestamp": timestamp,
        "best_cost": float(best_cost),
        "best_position_ordered": best_pos.tolist(),  
        "pso_hyperparameters": pso_options,
        "fixed_physical_parameters": sanitized_fp
    }
    
    with open(os.path.join(run_dir, "optimization_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
        
    # 2. Сохранение истории сходимости
    np.savez_compressed(
        os.path.join(run_dir, "convergence_history.npz"),
        cost_history=np.array(cost_history)
    )
    
    print(f"[Optimizer Logging] Сессия оптимизации сохранена в: {run_dir}")
    return run_dir
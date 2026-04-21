# заебало выдавать ошибку

"""

# 7. ОПТИМИЗАЦИЯ

# 1. Фиксированные параметры
fixed_params = {
    'r_ohm': 3.73e-4,           # Ом
    'A': 0.08,              # м
    'U_0': 500,             # В
    'N': 200,               # кол-во точек
    'grid': np.linspace(-0.2, 0.2, 200),  # сетка для оценки однородности
    'L_own': 0.21e-6,          # Гн
    'R': np.full(n, 0.035),    # м (радиус колец)
    'n': 5,                 # шт (кол-во колец в стопке)
    'm': 6,                 # шт (кол-во стопок)
    'delta': np.array([0.01, 0.001, 0.01, 0.001]),          # м (зазор между кольцами)
    'B_target': 10e-6,        # Целевая индукция (10 мкТл)
    'weight_magnitude': 0.3,  # Вес величины поля (0.3 = 30% важности)
    'B_min_threshold': 1e-6
}

# 2. Создание оптимизатора
optimizer = MRIOptimizer(fixed_params=fixed_params)

# 3. Границы для варьируемых параметров [C, omega, phi]
bounds = (
    [10e-12,   2*np.pi*50e6,   0.0],        # минимумы: C=10пФ, f=50МГц, phi=0
    [100e-12,  2*np.pi*100e6,  2*np.pi]    # максимумы: C=100пФ, f=100МГц, phi=2π
)

# 4. Параметры PSO
pso_options = {
    'c1': 0.5,
    'c2': 0.3,
    'w': 0.9,
}

# 5. Запуск оптимизации
best_pos, best_cost = optimizer.optimize(
    bounds=bounds,
    pso_options=pso_options,
    n_particles=10,
    max_iterations=10
)

# 6. Результаты
print("\n=== Оптимальные параметры ===")
print(f"Ёмкость:     {best_pos[0]:.2e} Ф ({best_pos[0]*1e12:.2f} пФ)")
print(f"Частота:     {best_pos[1]/(2*np.pi):.2f} МГц (omega={best_pos[1]:.2e} рад/с)")
print(f"Фаза:        {best_pos[2]:.4f} рад ({np.degrees(best_pos[2]):.2f}°)")
print(f"Однородность: {best_cost:.6f}")

print("Построение резонансной характеристики...")
frequencies = np.linspace(50e6, 80e6, 50)
B_center = []

# Уменьшаем дискретизацию для ускорения расчёта АЧХ
N_seg_ACH = 128  

for i, f in enumerate(frequencies):
    omega_test = 2 * np.pi * f
    Z_self_test = Z_self_matrix(
        r=r_ohm, C=C, n=n, m=m, R=radii, omega=omega_test
    )
    I_test = calc_I(
        Z_self=Z_self_test, U=U, omega=omega_test, L=L, n=n, m=m
    )
    B_test = b_s_l(
        obs_points=np.array([[0, 0, 0]]),
        I_matrix=I_test,
        N=N_seg_ACH,
        n=n,
        m=m,
        all_coordinates=all_coordinates
    )
    B_center.append(np.linalg.norm(B_test[0]))
    
    # Прогресс
    if (i + 1) % 10 == 0:
        print(f"  Частоты обработаны: {i + 1}/{len(frequencies)}")

B_center = np.array(B_center)

# Проверка на пустые данные
if np.all(np.isnan(B_center)):
    print(" Предупреждение: Все значения поля NaN!")
else:
    plot_resonance_curve(
        frequencies=frequencies,
        B_values=B_center,
        resonance_freq=frequency_MHz * 1e6,  
        save_path='resonance_curve.png'
    )
"""
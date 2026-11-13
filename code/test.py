# test_n_coarse.py
"""
Проверка: при каком n_coarse find_resonance_frequency даёт стабильный f_res.

Идея:
  1. Берём эталон: n_coarse = 500 (медленно, но точно).
  2. Для n_coarse = 5, 10, 20, 30, 50, 100, 200 -- считаем f_res.
  3. Сравниваем с эталоном. Смотрим, при каком n_coarse f_res совпадает.

Запуск:
    python test_n_coarse.py
"""

import json
import numpy as np

import src.config as config
from src.geometry import points_on_rings_general, all_rings_centers, stack_basis
from src.inductance_quasystat import inductance_matrix_quasystat, set_geometry
from src.resonance_find import find_resonance_frequency


# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------
N_COARSE_LIST = [5, 10, 20, 30, 50, 100, 200]
N_COARSE_REF  = 500                  # эталон

F_MIN = 250e6
F_MAX = 350e6
OBS_POINT = np.array([[0.0, 0.0, 0.0]])

# Сколько случайных геометрий проверить
N_GEOMETRIES = 5

# Фиксированные параметры
n = config.n
m = config.m
A = config.A
r_ohm = config.r_ohm
U_0 = config.U_0
L_own = config.L_self
C_val = float(np.asarray(config.C).flat[0])   # одна ёмкость для всех

print(f"Параметры: n={n}, m={m}, A={A}, C={C_val*1e12:.2f} пФ")
print(f"Диапазон: {F_MIN/1e6:.0f}–{F_MAX/1e6:.0f} МГц")
print(f"Эталон: n_coarse = {N_COARSE_REF}")
print()


# ---------------------------------------------------------------------------
# Функция: для данной геометрии -- f_res при разных n_coarse
# ---------------------------------------------------------------------------
def measure_f_res(radii, deltas, n_coarse):
    """Возвращает f_res для данной геометрии и n_coarse."""
    fi = 2 * np.pi / m
    _, normals = points_on_rings_general(
        delta=deltas, n=n, A=A, N=1024, R=radii, m=m
    )
    centers = all_rings_centers(delta=deltas, A=A, n=n, m=m, fi=fi)
    e1, e2 = stack_basis(m)

    set_geometry(
        m=m, n=n, ring_centers=centers,
        e1=e1, e2=e2, radii=radii, L_own=L_own,
    )
    L = inductance_matrix_quasystat(f_hz=300e6)

    C_array = np.full(n * m, C_val)

    f_res, _, _, peaks = find_resonance_frequency(
        L=L, n=n, m=m, radii=radii,
        A=A, gaps=deltas,
        C_value=C_array, r_ohm=r_ohm, U_0=U_0,
        ring_centers=centers, normals=normals,
        obs_point=OBS_POINT,
        f_min=F_MIN, f_max=F_MAX,
        n_coarse=n_coarse, n_fine=4000,
        aaa_tol=1e-8, aaa_mmax=80,
        prominence_frac=0.01,
        return_curve=True, return_all_peaks=True,
    )
    return f_res, len(peaks)


# ---------------------------------------------------------------------------
# Случайные геометрии
# ---------------------------------------------------------------------------
np.random.seed(42)   # для воспроизводимости

radii_list = []
deltas_list = []

for k in range(N_GEOMETRIES):
    # radii в разумных пределах: 0.02 ... 0.05
    radii = np.random.uniform(0.02, 0.05, n)
    # deltas: 0.001 ... 0.04, с проверкой на пересечение
    while True:
        deltas = np.random.uniform(0.001, 0.04, n - 1)
        # проверка: кольца в стопке не пересекаются
        ok = True
        for i in range(m):
            for j in range(i + 1, m):
                d = 2 * A * np.sin(np.pi * (j - i) / m)
                for ir in range(n):
                    for jr in range(n):
                        if radii[ir] + radii[jr] > d * 0.95:
                            ok = False
                            break
        if ok:
            break
    radii_list.append(radii)
    deltas_list.append(deltas)


# ---------------------------------------------------------------------------
# Основной тест
# ---------------------------------------------------------------------------
print("=" * 80)
print("ТЕСТ: f_res при разных n_coarse")
print("=" * 80)

# Таблица результатов
results = np.zeros((N_GEOMETRIES, len(N_COARSE_LIST) + 1))  # +1 для эталона

for k in range(N_GEOMETRIES):
    radii = radii_list[k]
    deltas = deltas_list[k]
    print(f"\n--- Геометрия {k+1}/{N_GEOMETRIES} ---")
    print(f"  radii  = {np.round(radii * 1e3, 1)} мм")
    print(f"  deltas = {np.round(deltas * 1e3, 1)} мм")

    # Эталон
    f_ref, n_peaks_ref = measure_f_res(radii, deltas, N_COARSE_REF)
    print(f"  эталон (n_coarse={N_COARSE_REF}): "
          f"f_res = {f_ref/1e6:.4f} МГц, пиков = {n_peaks_ref}")

    results[k, -1] = f_ref / 1e6  # эталон в МГц

    # Тестовые n_coarse
    for i, nc in enumerate(N_COARSE_LIST):
        f_res, n_peaks = measure_f_res(radii, deltas, nc)
        diff_khz = (f_res - f_ref) / 1e3
        results[k, i] = f_res / 1e6
        print(f"  n_coarse={nc:>3}: f_res = {f_res/1e6:.4f} МГц, "
              f"Δ = {diff_khz:+.1f} кГц, пиков = {n_peaks}")


# ---------------------------------------------------------------------------
# Сводка: насколько стабильно при каждом n_coarse
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("СВОДКА: |Δf| относительно эталона (n_coarse=500)")
print("=" * 80)

# Заголовок
header = f"{'n_coarse':>10} | " + " | ".join(
    f"геом {k+1}" for k in range(N_GEOMETRIES)
) + f" | {'макс |Δ|, кГц':>14}"
print(header)
print("-" * len(header))

for i, nc in enumerate(N_COARSE_LIST):
    diffs = np.abs(results[:, i] - results[:, -1]) * 1000  # в кГц
    row = f"{nc:>10} | " + " | ".join(
        f"{diffs[k]:>6.1f}" for k in range(N_GEOMETRIES)
    ) + f" | {diffs.max():>14.1f}"
    print(row)


# ---------------------------------------------------------------------------
# Рекомендация
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("РЕКОМЕНДАЦИЯ")
print("=" * 80)

THRESHOLD_KHZ = 100.0   # допустимое отклонение от эталона

for i, nc in enumerate(N_COARSE_LIST):
    diffs = np.abs(results[:, i] - results[:, -1]) * 1000
    if diffs.max() < THRESHOLD_KHZ:
        print(f"[OK] При n_coarse = {nc} отклонение < {THRESHOLD_KHZ} кГц "
              f"на всех {N_GEOMETRIES} геометриях.")
        print(f"     Можно использовать n_coarse = {nc}.")
        break
else:
    print(f"[WARN] Ни одно n_coarse из списка {N_COARSE_LIST} не даёт "
          f"отклонения < {THRESHOLD_KHZ} кГц.")
    print(f"       Надо больше точек или проверить AAA-настройки.")
import numpy as np
from src.geometry import points_on_rings_general, ring_center_general

# Параметры из main.py
m = 6
n = 5
A = 0.08
gaps = np.array([0.01, 0.001, 0.01, 0.001])
radii = np.full(n, 0.035)
N_seg = 512

# Расчёт геометрии
print("Расчёт геометрии системы...")
all_coordinates, normals = points_on_rings_general(
    delta=gaps,
    n=n,
    A=A,
    N=N_seg,
    R=radii,
    m=m
)

# Проверка форма normals
print(f"\nФорма normals: {normals.shape}, тип: {type(normals)}")
print(f"Первые 3 строки normals:")
print(normals[:3])
print()

# Проверка центров и нормалей
fi = 2 * np.pi / m
print(f"Проверка центров и нормалей (m={m}, n={n}):")
print(f"{'Стопка':<8} {'Кольцо':<8} {'Угол (°)':<12} {'Центр':<30} {'Нормаль':<35}")
print("-" * 100)
for stack_idx in range(m):
    angle = stack_idx * fi
    angle_deg = np.degrees(angle)
    for ring_idx in range(n):
        center = ring_center_general(gaps, A, n, fi, stack_idx, ring_idx)
        
        # Попытка получить нормаль разными способами
        if normals.shape[0] == m * n:
            normal = normals[stack_idx * n + ring_idx]
            idx_str = f"{stack_idx}*{n}+{ring_idx}"
        elif normals.shape[0] == m:
            normal = normals[stack_idx]
            idx_str = f"{stack_idx}"
        else:
            normal = None
            idx_str = "???"
        
        if normal is not None:
            center_str = f"[{center[0]:7.4f}, {center[1]:7.4f}, {center[2]:7.4f}]"
            normal_str = f"[{normal[0]:7.4f}, {normal[1]:7.4f}, {normal[2]:7.4f}]"
        else:
            center_str = "???"
            normal_str = "???"
            
        print(f"{stack_idx:<8} {ring_idx:<8} {angle_deg:<12.1f} {center_str:<30} {normal_str:<35}")

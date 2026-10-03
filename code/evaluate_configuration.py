import os
import json
import numpy as np
import matplotlib.pyplot as plt

import src.config as config
from src.geometry import points_on_rings_general, all_rings_centers
from src.currents import Z_self_matrix, generate_voltage_array, calc_I
from src.magnetic_field import b_s_l_optimized, quality_metric


from src.inductance_quasystat import inductance_matrix_quasystat
from src.inductance import inductance_matrix_nonqs





# парсить параметры из файла (чтобы визуализировать оптимизацию
LOAD_FROM_JSON = True
JSON_PATH = "optimization_results/opt_20260923_035658/optimized_config.json"


CUSTOM_C = 39.36e-12   # 39.36 пФ


CUSTOM_RADII = np.array([0.045, 0.040, 0.035, 0.035, 0.030])


CUSTOM_DELTAS = np.array([0.012, 0.008, 0.015, 0.005])


CALCULATION_MODE = 'quasistatic'


X_SPAN_MM = 60.0          # Диапазон построения поля
NUM_POINTS_X = 300        # Количество расчетных точек
ROI_FRACTION = 0.40       # (круг радиусом k*A)


RING_COLOR = 'tab:blue'


def main():


    # Загрузка параметров
    fp = config.get_fixed_params()
    m = int(fp['m'])
    A = float(fp['A'])
    omega = float(fp['omega'])
    r_ohm = float(fp['r_ohm'])
    U_0 = float(fp['U_0'])
    L_own = float(fp['L_own'])
    phi_0 = float(fp.get('phi', 0.0))

    if LOAD_FROM_JSON:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        C_val = float(data['C'])
        radii = np.array(data['radii'])
        deltas = np.array(data['deltas'])
        if 'm' in data: m = int(data['m'])
        if 'A' in data: A = float(data['A'])
    else:
        C_val = float(CUSTOM_C)
        radii = np.array(CUSTOM_RADII)
        deltas = np.array(CUSTOM_DELTAS)

    n = len(radii)
    if n > 1 and len(deltas) != n - 1:
        raise ValueError(f"Ошибка: для n={n} колец должно быть задано ровно {n-1} зазоров deltas!")

    fi = 2 * np.pi / m
    R_domain = ROI_FRACTION * A
    x_line = np.linspace(-X_SPAN_MM * 1e-3, X_SPAN_MM * 1e-3, NUM_POINTS_X)

    print(f"Стопок (m): {m}, колец в стопке (n): {n}")
    print(f"Емкость C:  {C_val * 1e12:.2f} пФ")
    print(f"Радиусы R:  {[round(r * 1e3, 2) for r in radii]} мм")
    if len(deltas) > 0:
        print(f"Зазоры d:   {[round(d * 1e3, 2) for d in deltas]} мм")
    print(f"Режим индуктивностей: {CALCULATION_MODE.upper()}")
    print("-" * 65)

    all_coords, normals = points_on_rings_general(
        delta=deltas, n=n, A=A, N=128, R=radii, m=m
    )
    global_centers = all_rings_centers(delta=deltas, A=A, n=n, m=m, fi=fi)
    stack_x_centers = np.cumsum(np.insert(deltas, 0, A)) if len(deltas) > 0 else np.array([A])



    if CALCULATION_MODE == 'quasistatic':
        L = inductance_matrix_quasystat(n=n, m=m, R=radii, L_own=L_own, A=A, delta=deltas)
    else:
        L = inductance_matrix_nonqs(n=n, m=m, R=radii, L_own=L_own, A=A, delta=deltas)



    C_array = np.full(n * m, C_val)
    Z_self = Z_self_matrix(r=r_ohm, C=C_array, n=n, m=m, R=radii, omega=omega)
    U = generate_voltage_array(U_0=U_0, m=m, n=n, phi_0=phi_0)
    I_matrix = calc_I(Z_self=Z_self, U=U, omega=omega, L=L, n=n, m=m)



    obs_points_x = np.stack([x_line, np.zeros_like(x_line), np.zeros_like(x_line)], axis=1)
    B_complex_x = b_s_l_optimized(obs_points_x, I_matrix, n, m, global_centers, normals, radii)
    B_amp_x = np.linalg.norm(B_complex_x, axis=1)


    cv_val = quality_metric(
        I=I_matrix, n=n, m=m, current_centers=global_centers,
        current_normals=normals, R_array=radii, R_domain=R_domain, N_grid=25
    )

    B_center = B_amp_x[len(B_amp_x) // 2]
    print(f"\nПоле в центре (X=0):   {B_center * 1e6:.2f} мкТл")
    print(f" Однородность CV (ROI): {cv_val * 100:.2f} % (std/mean)\n")


    coords_per_ring_mm = all_coords.reshape((m * n, 128, 3)) * 1000.0
    gc_mm = global_centers * 1000.0
    theta_circ = np.linspace(0, 2 * np.pi, 100)


    fig_3d = plt.figure(figsize=(9, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    for idx in range(m * n):
        pts = coords_per_ring_mm[idx]
        pts_closed = np.vstack([pts, pts[0]])
        label_ring = "Кольца катушки" if idx == 0 else ""
        ax_3d.plot(pts_closed[:, 0], pts_closed[:, 1], pts_closed[:, 2],
                   color=RING_COLOR, lw=1.6, alpha=0.85)

    ax_3d.scatter(gc_mm[:, 0], gc_mm[:, 1], gc_mm[:, 2], color='red', s=15, label=f"область расчета -- {ROI_FRACTION}*0.08 м")


    ax_3d.set_title("", fontsize=13, pad=15)
    ax_3d.set_xlabel("X (мм)", fontsize=11)
    ax_3d.set_ylabel("Y (мм)", fontsize=11)
    ax_3d.set_zlabel("Z (мм)", fontsize=11)
    ax_3d.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig("1_rings_3d.png", dpi=300)



    fig_xy, ax_xy = plt.subplots(figsize=(8, 8))

    for idx in range(m * n):
        pts = coords_per_ring_mm[idx]
        pts_closed = np.vstack([pts, pts[0]])
        label_ring = "Кольца катушки" if idx == 0 else ""
        ax_xy.plot(pts_closed[:, 0], pts_closed[:, 1], color=RING_COLOR, lw=1.4, alpha=0.8, label=label_ring)


    roi_patch = plt.Circle((0, 0), R_domain * 1000, color='lightgreen', alpha=0.35, label='Область ROI')
    ax_xy.add_patch(roi_patch)
    ax_xy.plot(R_domain * 1000 * np.cos(theta_circ), R_domain * 1000 * np.sin(theta_circ), 'g--', lw=1.5)


    ax_xy.plot([-X_SPAN_MM, X_SPAN_MM], [0, 0], color='tab:red', lw=2, linestyle='-', label='Ось X (профиль поля)')
    ax_xy.scatter(gc_mm[:, 0], gc_mm[:, 1], color='red', s=12, zorder=5)

    ax_xy.set_aspect('equal')
    ax_xy.set_title("Проекция катушки в плоскости XY (вид сверху)", fontsize=13)
    ax_xy.set_xlabel("X (мм)", fontsize=11)
    ax_xy.set_ylabel("Y (мм)", fontsize=11)
    ax_xy.grid(True, linestyle='--', alpha=0.5)
    ax_xy.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig("2_rings_xy_view.png", dpi=300)



    fig_stack, ax_stack = plt.subplots(figsize=(10, 5))
    ax_stack.axhline(0, color='gray', linestyle=':', lw=1.0)
    ax_stack.scatter([0], [0], color='black', marker='+', s=100, label='Центр системы (0,0)')


    for i, (xc, r) in enumerate(zip(stack_x_centers * 1000, radii * 1000)):
        ax_stack.plot([xc, xc], [-r, r], color=RING_COLOR, lw=3.5, solid_capstyle='round')
        ax_stack.text(xc, r + 2.0, f"R{i+1}={r:.1f}", ha='center', va='bottom',
                      fontsize=9, color=RING_COLOR, fontweight='bold')
        ax_stack.scatter([xc], [0], color='red', s=25)


    if len(deltas) > 0:
        for i, d in enumerate(deltas * 1000):
            x1 = stack_x_centers[i] * 1000
            x2 = stack_x_centers[i+1] * 1000
            y_arr = -max(radii * 1000) * 0.45
            ax_stack.annotate('', xy=(x1, y_arr), xytext=(x2, y_arr),
                              arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
            ax_stack.text((x1 + x2)/2, y_arr - 2.5, f"δ{i+1}={d:.1f}", ha='center', va='top', fontsize=8.5)


    ax_stack.annotate('', xy=(0, 0), xytext=(stack_x_centers[0] * 1000, 0),
                      arrowprops=dict(arrowstyle='<->', color='green', lw=1.2))
    ax_stack.text(stack_x_centers[0] * 500, 3.0, f"A={A*1000:.1f} мм",
                  ha='center', va='bottom', fontsize=9.5, color='green', fontweight='bold')

    ax_stack.set_title("Продольный чертеж стопки (размеры в мм)", fontsize=13)
    ax_stack.set_xlabel("Расстояние от центра вдоль стопки (мм)", fontsize=11)
    ax_stack.set_ylabel("Высота Z (мм)", fontsize=11)
    ax_stack.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("3_stack_profile.png", dpi=300)



    fig_b, ax_b = plt.subplots(figsize=(10, 5))
    x_mm = x_line * 1000.0
    B_uT = B_amp_x * 1e6  # в мкТл

    ax_b.plot(x_mm, B_uT, color='tab:red', lw=2.2)

    ax_b.axvspan(-R_domain * 1000, R_domain * 1000, color='lightgreen', alpha=0.3)
    ax_b.axvline(-R_domain * 1000, color='green', linestyle='--', lw=1.2)
    ax_b.axvline(R_domain * 1000, color='green', linestyle='--', lw=1.2)

    


    ax_b.set_title(f"|B| вдоль оси X, область расчета -- {ROI_FRACTION}*0.08 м", fontsize=13)
    ax_b.set_xlabel("X (мм)", fontsize=11)
    ax_b.set_ylabel("|B| (мкТл)", fontsize=11)
    ax_b.grid(True, linestyle='--', alpha=0.5)
    ax_b.legend(loc='lower center', fontsize=9.5)
    plt.tight_layout()
    plt.savefig("4_field_profile_X.png", dpi=300)



    plt.show()


if __name__ == '__main__':
    main()
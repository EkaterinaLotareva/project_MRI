import numpy as np
import matplotlib.pyplot as plt

def plot_field_contour(Phi, R, B_magnitude, C=None, title="", save_path="B_field.png"):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    contour = ax.contourf(Phi, R, B_magnitude, levels=200, cmap='jet', extend='both')
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('|B| (Тл)', rotation=270, labelpad=20)
    
    c_str = ""
    if C is not None:
        c_val = C[0] if hasattr(C, '__getitem__') else C
        c_str = f", C: {c_val * 1e12:.1f} pF"
    
    ax.set_title(title or f"B amplitude{c_str}", pad=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_rlabel_position(-22.5) 
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def plot_field_along_axis(axis_coords, B_amplitude, axis_name='X', title="", save_path="B_axis.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(axis_coords, B_amplitude, color='blue', lw=2, label='|B|')
    plt.xlabel(f'Координата {axis_name} (м)')
    plt.ylabel('|B| (Тл)')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

def visualize_rings(all_coords, ring_centers, normals, N_seg, n, m):
    """3D-визуализация геометрии всех колец катушки."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    coords_per_ring = all_coords.reshape((m * n, N_seg, 3))
    for ring_idx in range(m * n):
        pts = coords_per_ring[ring_idx]
        # замыкаем кольцо
        pts_closed = np.vstack([pts, pts[0]])
        ax.plot(pts_closed[:, 0], pts_closed[:, 1], pts_closed[:, 2], color='tab:blue', lw=1.5)

    ax.scatter(ring_centers[:, 0], ring_centers[:, 1], ring_centers[:, 2], color='red', s=15, label='Центры колец')
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')
    ax.set_zlabel('Z (м)')
    ax.set_title('Геометрия катушки')
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    import matplotlib.pyplot as plt

def plot_field_comparison(
    x_line, 
    B_amp_qs, 
    B_amp_nonqs, 
    axis_name='X',  
    save_path=None
):
    """Построение двух профилей поля и их разности."""
    
    # Создаём фигуру с двумя subplot'ами рядом
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ===== Левый график: сравнение профилей =====
    ax1.plot(x_line, B_amp_qs, 'b-', linewidth=2, label='Квазистат')
    ax1.plot(x_line, B_amp_nonqs, 'r-', linewidth=2, label='Неквазистат')
    
    ax1.set_xlabel(f'{axis_name} (м)', fontsize=11)
    ax1.set_ylabel('|B| (Тл)', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ===== Правый график: разность =====
    diff = np.asarray(B_amp_nonqs) - np.asarray(B_amp_qs)
    
    ax2.plot(x_line, diff, 'r-', linewidth=2)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle=':')
    
    ax2.set_xlabel(f'{axis_name} (м)', fontsize=11)
    ax2.set_ylabel('Δ|B| (Тл)', fontsize=11)
    ax2.set_title('Разность полей', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Опционально: относительная разность в процентах можно вывести в подписи
    max_abs_diff = np.max(np.abs(diff))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Сравнительный график сохранен: {save_path}")
    
    plt.show()
    return diff  # удобно возвращать разность для дальнейшего анализа


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_optimal_configuration(
    all_coords, global_centers, opt_radii, opt_deltas, 
    stack_x_centers, m, n, A, R_domain, opt_C, best_cost, 
    save_path="optimal_geometry.png"
):
    """
    Строит наглядную инфографику оптимальной геометрии:
    1. 3D вид всей катушки.
    2. Вид сверху (XY) с областью оценки однородности (ROI).
    3. Продольный чертеж стопки с размерами (R и delta в мм).
    """
    fig = plt.figure(figsize=(16, 8))
    
    # -------------------------------------------------------------
    # 1. 3D-МОДЕЛЬ СИСТЕМЫ КОЛЕЦ
    # -------------------------------------------------------------
    ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
    N_seg = all_coords.shape[0] // (m * n)
    coords_per_ring = all_coords.reshape((m * n, N_seg, 3))
    
    
    for s in range(m):
        for r_i in range(n):
            idx = s * n + r_i
            pts = coords_per_ring[idx] * 1000.0  # в мм
            pts_closed = np.vstack([pts, pts[0]])
            ax_3d.plot(pts_closed[:, 0], pts_closed[:, 1], pts_closed[:, 2], 
                       color='blue', lw=1.8, alpha=0.85)

    # Центры колец
    gc_mm = global_centers * 1000.0
    ax_3d.scatter(gc_mm[:, 0], gc_mm[:, 1], gc_mm[:, 2], color='black', s=20, label='Центры витков')
    
    # Сфера/диск области интереса ROI
    theta_roi = np.linspace(0, 2*np.pi, 100)
    ax_3d.plot(R_domain * 1000 * np.cos(theta_roi), 
               R_domain * 1000 * np.sin(theta_roi), 
               np.zeros_like(theta_roi), color='green', lw=2, linestyle='-')

    ax_3d.set_title("3D-геометрия катушки", fontsize=13, pad=15)
    ax_3d.set_xlabel("X (мм)")
    ax_3d.set_ylabel("Y (мм)")
    ax_3d.set_zlabel("Z (мм)")
    ax_3d.legend(loc='upper right', fontsize=9)

    # -------------------------------------------------------------
    # 2. ПРОЕКЦИЯ XY (ВИД СВЕРХУ)
    # -------------------------------------------------------------
    ax_xy = fig.add_subplot(2, 2, 2)
    for s in range(m):
        for r_i in range(n):
            idx = s * n + r_i
            pts = coords_per_ring[idx] * 1000.0
            pts_closed = np.vstack([pts, pts[0]])
            ax_xy.plot(pts_closed[:, 0], pts_closed[:, 1], color='blue', lw=1.2, alpha=0.7)
            
    # Область ROI в центре
    roi_circle = plt.Circle((0, 0), R_domain * 1000, color='lightgreen', alpha=0.35, label='Область DSV/ROI')
    ax_xy.add_patch(roi_circle)
    ax_xy.plot(R_domain * 1000 * np.cos(theta_roi), R_domain * 1000 * np.sin(theta_roi), 'g--', lw=1.5)

    ax_xy.scatter(gc_mm[:, 0], gc_mm[:, 1], color='red', s=12, zorder=5)
    ax_xy.set_aspect('equal')
    ax_xy.set_title("Проекция в плоскости XY (вид сверху)", fontsize=12)
    ax_xy.set_xlabel("X (мм)")
    ax_xy.set_ylabel("Y (мм)")
    ax_xy.grid(True, linestyle='--', alpha=0.5)
    ax_xy.legend(loc='upper right', fontsize=9)

    # -------------------------------------------------------------
    # 3. ЧЕРТЕЖ ПРОФИЛЯ СТОПКИ С РАЗМЕРАМИ
    # -------------------------------------------------------------
    ax_stack = fig.add_subplot(2, 2, 4)
    ax_stack.axhline(0, color='gray', linestyle=':', lw=1.0)
    
    # Начало стопки (луч от центра системы)
    ax_stack.scatter([0], [0], color='black', marker='+', s=100, label='Центр системы (0,0)')
    
    # Отрисовка витков как вертикальных сегментов высотой 2*R
    for i, (xc, r) in enumerate(zip(stack_x_centers * 1000, opt_radii * 1000)):
        # Вертикальная линия витка
        ax_stack.plot([xc, xc], [-r, r], color='tab:blue', lw=3.5, solid_capstyle='round')
        # Подпись радиуса
        ax_stack.text(xc, r + 2.0, f"R{i+1}={r:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='tab:blue')
        # Метка центра
        ax_stack.scatter([xc], [0], color='tab:red', s=25)

    # Отрисовка зазоров между витками
    if len(opt_deltas) > 0:
        for i, d in enumerate(opt_deltas * 1000):
            x_left = stack_x_centers[i] * 1000
            x_right = stack_x_centers[i+1] * 1000
            y_dim = -max(opt_radii * 1000) * 0.4
            
            # Размерная стрелка
            ax_stack.annotate(
                '', xy=(x_left, y_dim), xytext=(x_right, y_dim),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.0)
            )
            ax_stack.text((x_left + x_right)/2, y_dim - 2.5, f"δ{i+1}={d:.1f}", 
                          ha='center', va='top', fontsize=8, color='black')

    # Расстояние A от центра до первого витка
    ax_stack.annotate(
        '', xy=(0, 0), xytext=(stack_x_centers[0] * 1000, 0),
        arrowprops=dict(arrowstyle='<->', color='green', lw=1.2)
    )
    ax_stack.text(stack_x_centers[0] * 500, 3.0, f"A={A*1000:.1f} мм", 
                  ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')

    # Информационный блок с параметрами
    info_text = (
        f"Оптимальные параметры:\n"
        f"• CV (std/mean) = {best_cost * 100:.2f}%\n"
        f"• Емкость C = {opt_C * 1e12:.2f} пФ\n"
        f"• Стопок m = {m}, витков в стопке n = {n}\n"
        f"• ROI радиус = {R_domain * 1000:.1f} мм"
    )
    ax_stack.text(0.03, 0.95, info_text, transform=ax_stack.transAxes, 
                  va='top', ha='left', fontsize=9, 
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='linen', alpha=0.8, edgecolor='gray'))

    ax_stack.set_title("Продольный профиль стопки (размеры в мм)", fontsize=12)
    ax_stack.set_xlabel("Расстояние от центра вдоль стопки X (мм)")
    ax_stack.set_ylabel("Высота Z (мм)")
    ax_stack.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
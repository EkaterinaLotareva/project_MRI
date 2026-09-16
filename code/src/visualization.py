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
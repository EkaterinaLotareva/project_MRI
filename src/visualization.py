import numpy as np
import matplotlib.pyplot as plt

mu0 = 1




#Построение силовых линий

def xz_streamplot(R=1.0, I=1.0, x_extent=2.0, z_extent=2.0, Nx=200, Nz=200, Nseg=1024):
    xs = np.linspace(-x_extent, x_extent, Nx)
    zs = np.linspace(-z_extent, z_extent, Nz)
    X, Z = np.meshgrid(xs, zs, indexing='ij')
    Bx = np.zeros_like(X)
    Bz = np.zeros_like(Z)
    rp, dl = loop_segments(R, Nseg)
    for i in range(Nx):
        for k in range(Nz):
            r_obs = np.array([xs[i], 0.0, zs[k]])
            B = bio_savar_loop_point(r_obs, R=R, I=I, N=Nseg)
            Bx[k,i] = B[0]
            Bz[k,i] = B[2]
    speed = np.sqrt(Bx**2 + Bz**2)
    plt.figure(figsize=(6,6))
    strm = plt.streamplot(xs, zs, Bx, Bz, color=speed, cmap='plasma', density=1.5, linewidth=1) #Построение силовых линий магнитного поля
    plt.colorbar(label='|B| (T)') # Добавление цветовой шкалы для величины магнитного поля
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Линии магнитного поля в сечении y=0 (Bx, Bz)')
    plt.scatter([R, -R], [0, 0], color='black', s=10) # Отображение положения витка тока
    plt.axvline(0, color='k', linewidth=0.5) # Добавление оси симметрии
    plt.show()


def plot_field_contour(X, Y, B_magnitude, ring_centers=None, 
                       title=None, save_path=None, cmap='jet'):
    """
    Построение контурной карты магнитного поля с цветовой шкалой.
    
    Параметры:
    -----------
    X, Y : ndarray
        Координатные сетки (из np.meshgrid)
    B_magnitude : ndarray
        Величина магнитного поля |B| в каждой точке сетки
    ring_centers : list of array-like, optional
        Список центров колец для отображения на графике
    title : str, optional
        Заголовок графика
    save_path : str, optional
        Путь для сохранения изображения
    cmap : str, optional
        Цветовая схема (по умолчанию 'jet')
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Создаём контурный график и сохраняем объект ContourSet
    contour = ax.contourf(X, Y, B_magnitude, levels=50, cmap=cmap, vmax = 0.00125)
    
    # Добавляем цветовую шкалу с подписью
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('|B| (Тл)', rotation=270, labelpad=20, fontsize=10)
    
    # Отображаем центры колец (если переданы)
    if ring_centers is not None and len(ring_centers) > 0:
        ring_centers = np.array(ring_centers)
        ax.scatter(ring_centers[:, 0], ring_centers[:, 1], 
                  c='black', s=30, marker='o', edgecolors='white', 
                  linewidths=1.5, label='Кольца', zorder=5)
        ax.legend(loc='upper right', fontsize=8)
    
    # Оформление осей и сетки
    ax.set_xlabel('X (м)', fontsize=11)
    ax.set_ylabel('Y (м)', fontsize=11)
    ax.set_aspect('equal')  # Одинаковый масштаб осей
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    if title:
        ax.set_title(title, fontsize=12, pad=15)
    
    #  Сохранение и отображение
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранён: {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax, cbar  


def plot_resonance_curve(frequencies, B_values, resonance_freq=None,
                         xlabel='Частота (МГц)', ylabel='|B| (Тл)',
                         title=None, save_path=None):
    """Построение резонансной характеристики"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frequencies/1e6, B_values, 'b-', linewidth=2)
    if resonance_freq:
        ax.axvline(resonance_freq, color='red', linestyle='--', 
                  label=f'Рабочая частота = {resonance_freq:.1f} МГц')
    # ... остальной код
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    return fig, ax
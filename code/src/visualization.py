import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, PowerNorm, Normalize
from matplotlib.colors import AsinhNorm, LogNorm, Normalize
import matplotlib.ticker as ticker


def plot_field_contour(X, Y, B_magnitude, ring_centers=None, 
                       title=None, save_path=None, cmap='turbo'):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. Находим значение поля в центре (0, 0)
    # Ищем индекс точки, максимально близкой к началу координат
    dist_sq = X**2 + Y**2
    idx_center = np.unravel_index(np.argmin(dist_sq), X.shape)
    B_center = B_magnitude[idx_center]
    
    # 2. Устанавливаем границы шкалы
    v_min = 1e-7 # Чтобы "ноль" был синим
    v_max = B_center * 3 # Ваш новый лимит: удвоенное поле в центре
    
    # Если вдруг в центре 0 (бывает при специфичных модах), 
    # добавим проверку, чтобы код не упал
    if v_max <= v_min:
        v_max = np.max(B_magnitude) * 0.5

    # 3. Создаем уровни (levels) по логарифмической шкале
    levels = np.logspace(np.log10(v_min), np.log10(v_max), 100)
    
    # 4. Отрисовка
    # extend='both' закрасит все, что выше v_max, в темно-красный
    contour = ax.contourf(X, Y, B_magnitude, levels=levels, 
                          cmap=cmap, norm=LogNorm(vmin=v_min, vmax=v_max), 
                          extend='both')
    
    # 5. Настройка цветовой шкалы (Colorbar)
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('|B| (Тл)', rotation=270, labelpad=20)
    
    # Чтобы значения не пропадали, используем "умный" логарифмический локатор
    cbar.ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    cbar.ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    # 6. Оформление (остается прежним)
    if ring_centers is not None and len(ring_centers) > 0:
        rc = np.array(ring_centers)
        ax.scatter(rc[:, 0], rc[:, 1], c='black', s=25, edgecolors='white', 
                   linewidths=1, label='Кольца', zorder=5)
        ax.legend(loc='upper right', fontsize=9)

    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    if title:
        ax.set_title(title, fontsize=12, pad=15)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig, ax, cbar




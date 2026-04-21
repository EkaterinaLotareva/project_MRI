import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, PowerNorm, Normalize


def plot_field_contour(X, Y, B_magnitude, ring_centers=None, 
                       title=None, save_path=None, cmap='jet',
                       norm_method='asinh', percentile=99,
                       vmax=None, clip_outliers=True):
    """
    Построение контурной карты магнитного поля с цветовой шкалой.
    Использует asinh нормализацию по умолчанию для борьбы с сингулярностями.
    
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
    norm_method : str, optional
        Метод нормализации: 'asinh', 'percentile', 'symlog', 'power', 'linear'
        По умолчанию 'asinh' - лучше всего справляется с сингулярностями
    percentile : float, optional
        Процентиль для отсечения выбросов (по умолчанию 99)
    vmax : float, optional
        Явное указание максимума для нормализации (переопределяет percentile)
    clip_outliers : bool, optional
        Агрессивно отсекать выбросы (по умолчанию True)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Подготовка данных и нормализация для избежания сингулярностей
    B_data = B_magnitude.copy()
    B_finite = B_data[np.isfinite(B_data)]
    
    # Применяем метод нормализации
    if norm_method == 'asinh':
        # Гиперболический синус нормализация (мягче чем symlog, лучше для сингулярностей)
        B_data = np.asinh(B_data / np.percentile(B_finite, 50))
        norm = Normalize(vmin=0, vmax=np.percentile(B_data[np.isfinite(B_data)], 95))
    elif norm_method == 'percentile':
        # Отсекаем выбросы на основе процентиля
        if vmax is None:
            vmax = np.percentile(B_finite, percentile)
        B_data = np.clip(B_data, 0, vmax)
        norm = Normalize(vmin=0, vmax=vmax)
    elif norm_method == 'symlog':
        # Логарифмическая нормализация для широкого диапазона значений
        if vmax is None:
            vmax = np.percentile(B_finite, percentile)
        norm = SymLogNorm(linthresh=1e-6, vmin=B_data.min(), vmax=vmax)
    elif norm_method == 'power':
        # Степенная нормализация (gamma=0.5 подчеркивает детали)
        if vmax is None:
            vmax = np.percentile(B_finite, percentile)
        B_data = np.clip(B_data, 0, vmax)
        norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    else:  # 'linear'
        if vmax is None:
            vmax = np.percentile(B_finite, percentile)
        B_data = np.clip(B_data, 0, vmax)
        norm = Normalize(vmin=0, vmax=vmax)
    
    # Создаём контурный график и сохраняем объект ContourSet
    contour = ax.contourf(X, Y, B_data, levels=50, cmap=cmap, norm=norm)
    
    # Добавляем цветовую шкалу с подписью
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('|B| (Тл)', rotation=270, labelpad=20, fontsize=10)
    
    # Отображаем центры колец 
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


def analyze_field_distribution(B_magnitude, title=""):
    """
    Анализ распределения магнитного поля для выбора оптимального метода нормализации.
    
    Параметры:
    -----------
    B_magnitude : ndarray
        Величина магнитного поля
    title : str, optional
        Описание данных
    
    Возвращает:
    -----------
    dict : Словарь с статистикой и рекомендациями
    """
    B_data = B_magnitude.copy()
    B_finite = B_data[np.isfinite(B_data)]
    
    # Вычисляем статистику
    stats = {
        'min': B_finite.min(),
        'max': B_finite.max(),
        'mean': B_finite.mean(),
        'median': np.median(B_finite),
        'std': B_finite.std(),
        'p99': np.percentile(B_finite, 99),
        'p95': np.percentile(B_finite, 95),
        'p90': np.percentile(B_finite, 90),
    }
    
    # Коэффициент вариации
    cv = stats['std'] / stats['mean'] if stats['mean'] != 0 else 0
    
    # Рекомендация метода нормализации
    ratio = stats['max'] / stats['p99'] if stats['p99'] != 0 else 1
    
    print(f"\n{'='*60}")
    print(f"📊 Анализ распределения поля {title}")
    print(f"{'='*60}")
    print(f"Min |B|:        {stats['min']:.3e} Тл")
    print(f"Max |B|:        {stats['max']:.3e} Тл")
    print(f"Mean |B|:       {stats['mean']:.3e} Тл")
    print(f"Median |B|:     {stats['median']:.3e} Тл")
    print(f"Std |B|:        {stats['std']:.3e} Тл")
    print(f"Коэф. вариации: {cv:.2f}")
    print(f"\nПроцентили:")
    print(f"  P90:  {stats['p90']:.3e} Тл")
    print(f"  P95:  {stats['p95']:.3e} Тл")
    print(f"  P99:  {stats['p99']:.3e} Тл")
    print(f"  Max:  {stats['max']:.3e} Тл")
    print(f"\nОтношение Max/P99: {ratio:.2f}x")
    
    # Анализ выбросов
    q1 = np.percentile(B_finite, 25)
    q3 = np.percentile(B_finite, 75)
    iqr = q3 - q1
    outliers = np.sum((B_finite > q3 + 1.5*iqr) | (B_finite < q1 - 1.5*iqr))
    
    print(f"\nВыбросы (IQR метод): {outliers} точек ({100*outliers/len(B_finite):.2f}%)")
    
    # Рекомендация
    print(f"\n💡 Рекомендация:")
    if ratio > 10:
        print(f"  ✓ Используйте 'asinh' (сингулярности очень сильные)")
        rec_method = 'asinh'
    elif cv > 2:
        print(f"  ✓ Используйте 'symlog' (широкий диапазон значений)")
        rec_method = 'symlog'
    else:
        print(f"  ✓ Используйте 'linear' или 'power' (равномерное распределение)")
        rec_method = 'power'
    
    stats['recommended_method'] = rec_method
    stats['outlier_count'] = outliers
    
    return stats


def plot_field_distribution(B_magnitude, title="Распределение магнитного поля", 
                           save_path=None):
    """
    Визуализация распределения значений магнитного поля в виде гистограммы.
    
    Параметры:
    -----------
    B_magnitude : ndarray
        Величина магнитного поля
    title : str, optional
        Заголовок
    save_path : str, optional
        Путь для сохранения
    """
    B_finite = B_magnitude[np.isfinite(B_magnitude)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Обычная гистограмма
    axes[0].hist(B_finite, bins=100, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('|B| (Тл)', fontsize=11)
    axes[0].set_ylabel('Количество точек', fontsize=11)
    axes[0].set_title('Гистограмма: линейная шкала', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Логарифмическая гистограмма
    axes[1].hist(B_finite, bins=100, color='green', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('|B| (Тл)', fontsize=11)
    axes[1].set_ylabel('Количество точек (логарифмическая шкала)', fontsize=11)
    axes[1].set_title('Гистограмма: логарифмическая шкала Y', fontsize=11, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, which='both')
    
    fig.suptitle(title, fontsize=13, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранён: {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes

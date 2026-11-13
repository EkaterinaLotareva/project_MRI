# test_resonance.py
"""
Проверка find_resonance_frequency.

Запуск:
    python test_resonance.py
"""

import numpy as np
import matplotlib.pyplot as plt

import src.config as config
from src.geometry import points_on_rings_general, all_rings_centers
from src.inductance_quasystat import inductance_matrix_quasystat
from src.resonance_find import find_resonance_frequency


# --- параметры ---
n, m = config.n, config.m
radii, gaps, A = config.radii, config.gaps, config.A
r_ohm, U_0 = config.r_ohm, config.U_0
C_value = 23.7e-12

F_MIN, F_MAX = 30e6, 90e6
OBS_POINT = np.array([[0.0, 0.0, 0.0]])

# --- геометрия ---
all_coords, normals = points_on_rings_general(
    delta=gaps, n=n, A=A, N=config.N_seg, R=radii, m=m
)
ring_centers = all_rings_centers(gaps, A, n, m, 2 * np.pi / m)

print(f"[geom] n={n}, m={m}, колец = {n*m}")
print(f"[geom] ring_centers.shape = {ring_centers.shape}")
print(f"[geom] normals.shape = {normals.shape}")
from src.geometry import stack_basis
from src.inductance_quasystat import set_geometry

e1, e2 = stack_basis(m)
set_geometry(
    m=m, n=n,
    ring_centers=ring_centers,
    e1=e1, e2=e2,
    radii=radii,
    L_own=config.L_self,
)
L = inductance_matrix_quasystat(f_hz=F_MIN)   # частота в Гц (30 МГц)
print(f"[L] shape = {L.shape}")
print()
print(f"[L] shape = {L.shape}")
print()

# --- поиск ---
print("=" * 60)
print("find_resonance_frequency")
print("=" * 60)

f_res, f_fine, B_fine, peaks_freqs = find_resonance_frequency(
    L=L, n=n, m=m, radii=radii, A=A, gaps=gaps,
    C_value=C_value, r_ohm=r_ohm, U_0=U_0,
    ring_centers=ring_centers, normals=normals,
    obs_point=OBS_POINT,
    f_min=F_MIN, f_max=F_MAX,
    n_coarse=300, n_fine=8000,
    aaa_tol=1e-9, aaa_mmax=100,
    prominence_frac=0.01,
    return_curve=True, return_all_peaks=True,
)

print(f"\nВыбранный пик (самый правый): f_res = {f_res/1e6:.4f} МГц")
print(f"Все пики ({len(peaks_freqs)} шт):")
for f in peaks_freqs:
    print(f"  {f/1e6:.4f} МГц")
print()

# --- проверки ---
print("=" * 60)
print("ПРОВЕРКИ")
print("=" * 60)

ok = True

inside = F_MIN <= f_res <= F_MAX
print(f"[1] f_res внутри [{F_MIN/1e6}, {F_MAX/1e6}] МГц: {inside}")
ok &= inside

if len(peaks_freqs) > 0:
    is_rightmost = np.isclose(f_res, peaks_freqs[-1], rtol=1e-8, atol=1.0)
    print(f"[2] f_res == самый правый пик: {is_rightmost}")
    ok &= is_rightmost

finite = np.all(np.isfinite(B_fine))
print(f"[3] B_fine без NaN/Inf: {finite}")
ok &= finite

n_peaks = len(peaks_freqs)
print(f"[4] пиков найдено: {n_peaks}")
if n_peaks < 2:
    print("    FAIL: меньше 2 пиков")
    ok = False

# информационная
mag_fine = np.linalg.norm(B_fine, axis=1)
mag_at_res = mag_fine[np.argmin(np.abs(f_fine - f_res))]
mag_max = mag_fine.max()
print(f"[5] (инфо) |B| на резонансе = {mag_at_res:.3e}, "
      f"максимум = {mag_max:.3e}, отношение = {mag_at_res/mag_max:.3f}")

print()
print("ИТОГ:", "PASS" if ok else "FAIL")


# --- график ---
plt.figure(figsize=(12, 5))
plt.plot(f_fine/1e6, mag_fine, 'b-', lw=1.5, label='|B| (AAA)')
plt.plot(peaks_freqs/1e6,
         np.interp(peaks_freqs, f_fine, mag_fine),
         'ro', ms=8, label=f'пики ({n_peaks})')
plt.axvline(f_res/1e6, color='k', ls='--', lw=2,
            label=f'выбран: {f_res/1e6:.3f} МГц')
plt.yscale('log')
plt.xlabel('Частота, МГц')
plt.ylabel(r'$|B|$ в центре, Тл')
plt.title('Поиск резонанса по |B| в центре (векторный модуль, AAA)')
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
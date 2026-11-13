import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import src.config as config
from src.geometry import points_on_rings_general, all_rings_centers, stack_basis
from src.inductance_quasystat import mutual_inductance_inclined_fast
from src.nonqs_inductance import mutual_inductance_rings_nonqs
from src.currents import Z_self_matrix, generate_voltage_array, calc_I
from src.magnetic_field import b_s_l_optimized


# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------
FREQ_MIN_HZ = 200e6
FREQ_MAX_HZ = 400e6
N_POINTS    = 10000
OBS_POINT   = np.array([[0.0, 0.0, 0.0]])

_N_LGRID = 2
f_Lgrid  = np.linspace(FREQ_MIN_HZ, FREQ_MAX_HZ, _N_LGRID)


# ---------------------------------------------------------------------------
# Геометрия
# ---------------------------------------------------------------------------
n, m = config.n, config.m
radii, gaps, A = config.radii, config.gaps, config.A
r_ohm, U_0, C = config.r_ohm, config.U_0, config.C
L_own = config.L_self

all_coords, normals = points_on_rings_general(
    delta=gaps, n=n, A=A, N=config.N_seg, R=radii, m=m
)
ring_centers = all_rings_centers(gaps, A, n, m, 2 * np.pi / m)
e1, e2 = stack_basis(m)
U_const = generate_voltage_array(U_0=U_0, m=m, n=n)


# ---------------------------------------------------------------------------
# L_qs — быстро
# ---------------------------------------------------------------------------
def build_L_qs():
    mn = m * n
    L = np.zeros((mn, mn), dtype=complex)
    for i in range(mn):
        i_s, i_r = i // n, i % n
        for j in range(i, mn):
            j_s, j_r = j // n, j % n
            if i == j:
                L[i, j] = L_own
                continue
            M = mutual_inductance_inclined_fast(
                R1=radii[i_r], R2=radii[j_r],
                c1=ring_centers[i], e1a=e1[i_s], e1b=e2[i_s],
                c2=ring_centers[j], e2a=e1[j_s], e2b=e2[j_s],
            )
            L[i, j] = M
            L[j, i] = M
    return L


def build_L_nonqs_at_f(f_hz):
    mn = m * n
    L = np.zeros((mn, mn), dtype=complex)
    for i in range(mn):
        i_s, i_r = i // n, i % n
        for j in range(i, mn):
            j_s, j_r = j // n, j % n
            if i == j:
                L[i, j] = L_own
                continue
            M = mutual_inductance_rings_nonqs(
                c1=ring_centers[i], e1a=e1[i_s], e1b=e2[i_s], r1=radii[i_r],
                c2=ring_centers[j], e2a=e1[j_s], e2b=e2[j_s], r2=radii[j_r],
                f=f_hz, epsabs=1e-8, epsrel=1e-6,
            )
            L[i, j] = M
            L[j, i] = M
    return L


print("Расчёт L_qs...")
t0 = time.time()
L_qs = build_L_qs()
print(f"  {time.time()-t0:.2f} с")

print(f"Расчёт L_nonqs на {_N_LGRID} опорных частотах...")
t0 = time.time()
L_nonqs_stack = np.array([build_L_nonqs_at_f(f) for f in f_Lgrid])
print(f"  {time.time()-t0:.2f} с")


def L_nonqs_at(f_hz):
    idx = np.searchsorted(f_Lgrid, f_hz)
    idx = np.clip(idx, 1, _N_LGRID - 1)
    f0, f1 = f_Lgrid[idx - 1], f_Lgrid[idx]
    w = (f_hz - f0) / (f1 - f0)
    return (1 - w) * L_nonqs_stack[idx - 1] + w * L_nonqs_stack[idx]


# ---------------------------------------------------------------------------
# Отклик
# ---------------------------------------------------------------------------
def run_at_frequency(f_hz, L):
    omega = 2 * np.pi * f_hz
    Z = Z_self_matrix(r=r_ohm, C=C, n=n, m=m, R=radii, omega=omega)
    I = calc_I(Z_self=Z, U=U_const, omega=omega, L=L, n=n, m=m)
    B = b_s_l_optimized(OBS_POINT, I, n, m, ring_centers, normals, radii)
    return B[0, 0]


# ---------------------------------------------------------------------------
# Цикл по частоте
# ---------------------------------------------------------------------------
freqs = np.linspace(FREQ_MIN_HZ, FREQ_MAX_HZ, N_POINTS)
B_qs    = np.zeros(N_POINTS, dtype=complex)
B_nonqs = np.zeros(N_POINTS, dtype=complex)

print(f"Расчёт {N_POINTS} точек...")
t0 = time.time()
for k, f in enumerate(freqs):
    B_qs[k]    = run_at_frequency(f, L_qs)
    B_nonqs[k] = run_at_frequency(f, L_nonqs_at(f))
print(f"  {time.time()-t0:.2f} с")


# опорные точки NonQS
B_nonqs_at_ref = np.array([run_at_frequency(f, L_nonqs_at(f)) for f in f_Lgrid])
ref_vals = np.abs(B_nonqs_at_ref)


# ---------------------------------------------------------------------------
# Поиск пиков и параметры
# ---------------------------------------------------------------------------
mags_qs  = np.abs(B_qs)
mags_nqs = np.abs(B_nonqs)
phase_qs_deg    = np.degrees(np.unwrap(np.angle(B_qs)))
phase_nonqs_deg = np.degrees(np.unwrap(np.angle(B_nonqs)))


def find_all_peaks(freqs, mags, prominence_frac=0.01):
    peaks, _ = find_peaks(mags, prominence=prominence_frac * mags.max(), distance=3)
    return peaks


def analyze_peak(freqs, mags, phase_deg, idx_peak):
    f_peak = freqs[idx_peak]
    mag_peak = mags[idx_peak]
    phase_at_peak = phase_deg[idx_peak]

    half_level = mag_peak / np.sqrt(2.0)
    left = idx_peak
    while left > 0 and mags[left] > half_level:
        left -= 1
    right = idx_peak
    while right < len(mags) - 1 and mags[right] > half_level:
        right += 1

    if left == 0 or right == len(mags) - 1:
        fwhm = np.nan
        Q = np.nan
    else:
        f_left = freqs[left] + (freqs[left+1] - freqs[left]) * \
                 (half_level - mags[left]) / (mags[left+1] - mags[left])
        f_right = freqs[right-1] + (freqs[right] - freqs[right-1]) * \
                  (half_level - mags[right-1]) / (mags[right] - mags[right-1])
        fwhm = f_right - f_left
        Q = f_peak / fwhm if fwhm > 0 else np.nan

    return {'f': f_peak, 'mag': mag_peak, 'phase': phase_at_peak,
            'fwhm': fwhm, 'Q': Q, 'idx': idx_peak}


def match_peaks(peaks_qs, peaks_nqs, freqs, max_shift_hz=5e6):
    pairs = []
    used_nqs = set()
    for iq in peaks_qs:
        f_q = freqs[iq]
        best_j = None
        best_dist = np.inf
        for jn in peaks_nqs:
            if jn in used_nqs:
                continue
            dist = abs(freqs[jn] - f_q)
            if dist < best_dist:
                best_dist = dist
                best_j = jn
        if best_j is not None and best_dist <= max_shift_hz:
            pairs.append((iq, best_j))
            used_nqs.add(best_j)
        else:
            pairs.append((iq, None))
    only_nqs = [jn for jn in peaks_nqs if jn not in used_nqs]
    return pairs, only_nqs


peaks_qs  = find_all_peaks(freqs, mags_qs,  0.01)
peaks_nqs = find_all_peaks(freqs, mags_nqs, 0.01)
analysis_qs  = [analyze_peak(freqs, mags_qs,  phase_qs_deg,    p) for p in peaks_qs]
analysis_nqs = [analyze_peak(freqs, mags_nqs, phase_nonqs_deg, p) for p in peaks_nqs]
pairs, only_nqs = match_peaks(peaks_qs, peaks_nqs, freqs)

# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax, logy in zip(axes, [False, True]):
    ax.plot(freqs/1e6, mags_qs,  'b-', lw=2, label='Квазистат.')
    ax.plot(freqs/1e6, mags_nqs, 'r-', lw=2, label='Неквазистат.')

    ax.plot(freqs[peaks_qs]/1e6,  mags_qs[peaks_qs],  'bd', ms=10,
            mec='k', mew=1, label='Пики квазистат', zorder=5)
    ax.plot(freqs[peaks_nqs]/1e6, mags_nqs[peaks_nqs], 'r^', ms=10,
            mec='k', mew=1, label='Пики неквазистат', zorder=5)

    for p in peaks_qs:
        ax.axvline(freqs[p]/1e6, color='b', ls='--', alpha=0.3, lw=1)
    for p in peaks_nqs:
        ax.axvline(freqs[p]/1e6, color='r', ls=':',  alpha=0.3, lw=1)


    ax.set_xlabel('Частота, МГц', fontsize=12)
    ax.set_ylabel(r'$|B|$ в центре, Тл', fontsize=12)
    ax.set_title('АЧХ' + (' ' if logy else ''), fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower left', fontsize=10)
    if logy:
        ax.set_yscale('log')

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------
# Печать в консоль с добротностью
# ---------------------------------------------------------------------------
def print_peaks_with_Q(freqs, mags, phase_deg, label):
    peaks, _ = find_peaks(mags, prominence=0.01 * mags.max(), distance=3)
    print(f"\n{label}:")
    print(f"  {'f, МГц':>10}  {'|B|, Тл':>12}  {'phase, °':>10}  "
          f"{'FWHM, МГц':>10}  {'Q':>10}")
    print("  " + "-" * 62)
    results = []
    for p in peaks:
        a = analyze_peak(freqs, mags, phase_deg, p)
        fwhm_mhz = a['fwhm'] / 1e6 if np.isfinite(a['fwhm']) else np.nan
        q_str = f"{a['Q']:.2f}" if np.isfinite(a['Q']) else "  nan"
        fwhm_str = f"{fwhm_mhz:10.4f}" if np.isfinite(fwhm_mhz) else "       nan"
        print(f"  {a['f']/1e6:10.4f}  {a['mag']:12.4e}  {a['phase']:10.2f}  "
              f"{fwhm_str}  {q_str:>10}")
        results.append(a)
    return results


print_peaks_with_Q(freqs, mags_qs,  phase_qs_deg,    "Пики QS")
print_peaks_with_Q(freqs, mags_nqs, phase_nonqs_deg, "Пики NonQS")
import numpy as np
import matplotlib.pyplot as plt

mu0 = 1

#Функция для дискретизация контура с током
def loop_segments(R=1.0, N=512):
    phi = np.linspace(0, 2*np.pi, N, endpoint=False) #Массив углов ϕ создается от 0 до 2π, endpoint=False исключает последнюю точку 2π
    dphi = 2*np.pi / N #Рассчитывается угловая длина ∆ϕ = 2π/N, необходимая для преобразования производной в конечный вектор сегмента
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = np.zeros_like(x)
    rp = np.stack((x, y, z), axis=1) #Создается массив центров сегментов rj′ = (xj, yj, zj) путем объединения x, y, z координат вдоль новой оси (axis=1).
    dl = np.stack((-R * np.sin(phi), R * np.cos(phi), np.zeros_like(phi)), axis=1) * dphi #Вычисляется вектор длины сегмента
    return rp, dl



#Б-С-Л для кольца с током

def bio_savar_loop_point(r_obs, R=1.0, I=1.0, N=1024):
    rp, dl = loop_segments(R, N)
    Rvecs = r_obs.reshape(1,3) - rp # массив векторов, направленных от каждого сегмента кольца к точке наблюдения
    norms = np.linalg.norm(Rvecs, axis=1)
    norms = np.where(norms < 1e-12, 1e-12, norms) #избегание деления на 0
    B = (mu0 / (4*np.pi)) * I * np.sum(np.cross(dl, Rvecs) / (norms**3).reshape(-1,1), axis=0)
    return B



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

#Рассчет поля соленоида
def B_solenoid(r_obs, R=1.0, I=1.0, n=10, L=1.0, Nphi=1024):
    r_obs = np.asarray(r_obs, dtype=float).reshape(3)
    Nturns = L*n
    x, y, z_obs = r_obs
    if Nturns <= 0:
        return np.zeros(3)
    if L <= 0:
        raise ValueError("length must be positive")

    z_positions = np.linspace(-L / 2.0, L / 2.0, int(Nturns))
    B = np.zeros(3, dtype=float)

    for zp in z_positions:

        r_rel = np.array([x, y, z_obs - zp], dtype=float)
        B_ring = bio_savar_loop_point(r_rel, R=R, I=I, N=Nphi)
        B += B_ring


    return B
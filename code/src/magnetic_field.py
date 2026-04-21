import numpy as np


mu0=4*np.pi*1e-7
mu0_over_4pi = mu0 / (4 * np.pi) 


#Функция для дискретизация контура с током
def loop_segments(R=1.0, N=512):
    phi = np.linspace(0, 2*np.pi, N, endpoint=False) #Массив углов ϕ создается от 0 до 2pi, endpoint=False исключает последнюю точку 2π
    dphi = 2*np.pi / N #Рассчитывается угловая длина, необходимая для преобразования производной в конечный вектор сегмента
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




mu0 = 4 * np.pi * 1e-7
mu0_over_4pi = mu0 / (4 * np.pi)


def b_s_l(obs_points, I_matrix, N_seg, n, m, all_coordinates, normals):
    N_obs = len(obs_points)
    B_total = np.zeros((N_obs, 3), dtype=complex)
    
    dtheta = 2 * np.pi / N_seg 
    
    for stack in range(m):
        normal = normals[stack]
        
        for ring in range(n):
            ring_glob = stack * n + ring
            seg_start = ring_glob * N_seg
            seg_end = seg_start + N_seg
            ring_points = all_coordinates[seg_start:seg_end]
            
            current = I_matrix[stack, ring]
            ring_center = np.mean(ring_points, axis=0)
            
            
            rad = ring_points - ring_center
            R = np.mean(np.linalg.norm(rad, axis=1))
            

            dl = np.cross(normal, rad)
            dl_norms = np.linalg.norm(dl, axis=1, keepdims=True)
            dl = dl / dl_norms
            
            dl_length = R * dtheta  # ← Из loop_segments
            dl_vecs = dl * dl_length
            
            for seg in range(N_seg):
                dl_vec = dl_vecs[seg]
                seg_pos = ring_points[seg]
                
                r_vecs = obs_points - seg_pos
                r_dist = np.linalg.norm(r_vecs, axis=1, keepdims=True)
            
                r_dist = np.where(r_dist < 1e-12, 1e-12, r_dist)
                
                cross = np.cross(dl_vec, r_vecs)
                dB = (mu0 / (4 * np.pi)) * current * cross / (r_dist**3)
                B_total += dB
    
    return B_total

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
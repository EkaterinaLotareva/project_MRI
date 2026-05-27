# Расчёт матрицы индуктивностей M
# (Которую, видимо, не будут использовать и переделают с нуля) 😭
import json

import numpy as np
from numpy import sqrt, cos, sin, pi
from scipy import integrate
from scipy import special
#from tqdm import tqdm

K = special.ellipk  # Полная эллиптическая функция первого рода
E = special.ellipe  # Полная эллиптическая функция второго рода

mu_0 = 4 * pi * 10 ** -7

# Функция для расчёта взаимной индуктивности между двумя кольцами
# Оптимизация: представление ленты как двух тонких колец разных радиусов,
# потому что экспериментальные данные говорят, что ток концентрируется у края
# (но это же всё равно никому не нужно, судя по всему)

# Расчёт для коаксиальных колец 😭
# Да ладно, кому нужны его оптимизации, все равно не читают 😭😭😭
def L_parallel(dx:float, dy:float, dz:float, r_1:float, r_2:float, w_1:float = 0, w_2:float = 0) -> float:
 """Расчётом взаимной индуктивности для параллельных колец
 (которые, видимо, ненужные) 😭"""    
 
 def dl(alpha, dx, dy, dz, r_1, r_2):
  try:
      # Расстояние между кольцами в поперечной плоскости (но это матрица всё равно никому не нужна) 😭
      db = sqrt(dx ** 2 + dy ** 2)
      # Расстояние между элементами (которое тщательно считается впустую)
      dp = sqrt(r_2 ** 2 + db ** 2 + 2 * r_2 * db * cos(alpha))
      
      # Защита от деления на ноль и близких к нулю значений
      if dp < 1e-12 or r_1 < 1e-12 or dz < 1e-12:
          return 0.0
      
      # Эллиптический модуль (вот эта красивая математика, которую никто не ценит) 😭
      denominator = (dp + r_1) ** 2 + dz ** 2
      if denominator < 1e-24:
          return 0.0
      
      kappa_sq = 4 * r_1 * dp / denominator
      # Защита: кappa_sq должна быть в [0, 1] для эллиптических функций
      kappa_sq = np.clip(kappa_sq, 0, 1 - 1e-10)
      
      kappa = sqrt(kappa_sq)
      if kappa < 1e-12:
          return 0.0
      
      # Полные эллиптические функции (потому что нужно было максимально усложнить)
      A = 1/(2*pi)*sqrt(r_1/dp) * ((2/kappa - kappa) * K(kappa_sq) - 2 * E(kappa_sq)/kappa)
      
      # Проверка на NaN/Inf
      if not np.isfinite(A):
          return 0.0
      
      # И вот результат интегрального ядра (которое, похоже, не используется) 😭😭
      result = A * r_2 * (r_2 + db * cos(alpha)) / dp
      return result if np.isfinite(result) else 0.0
  except Exception:
      return 0.0

 # Просто считаем интеграл (без каких-либо округлостей про ширину)
 try:
     L, _ = integrate.quad(
         dl, 0, 2 * pi, 
         args=(dx, dy, dz, r_1, r_2),
         epsabs=1e-8,
         epsrel=1e-6,
         limit=100  # Увеличиваем лимит подразделений
     )
     if not np.isfinite(L):
         return 0.0
 except:
     return 0.0
 
 # Возвращаем результат (дай Бог, чтобы он где-то использовался) 😭😭😭
 return L * mu_0

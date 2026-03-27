import numpy as np
from scipy.optimize import minimize
from tabulate import tabulate
import time

def model(t, x1, x2, x3, x4):
    return x1 + x2*t + x3*(t**2) + x4*(t**3)

def objetivo_ols(x, t, y):
    return 0.5 * np.sum((model(t, *x) - y) ** 2)

# ─── Carga de datos ───────────────────────────────────────────────────────────
data   = np.loadtxt("txt/data.txt")
t_all  = data[:, 0]
y_all  = data[:, 1]

x0     = [-1.0, -2.0, 1.0, -1.0]
bounds = [(-10, 10)] * 4

# ─── Tabla OLS para o = 0..15 ─────────────────────────────────────────────────
# Para cada o:
#   1. Ajustar OLS sobre todos los datos para identificar los o residuos más grandes
#   2. Eliminar esos o puntos (outliers detectados)
#   3. Ajustar OLS sobre los datos restantes y reportar f_ols(x*), #it, #fcnt, T(s)

rows = []
for o in range(16):

    # Paso 1: OLS completo para detectar qué puntos son los o outliers
    res_full = minimize(objetivo_ols, x0, args=(t_all, y_all),
                        method='L-BFGS-B', bounds=bounds)
    residuos  = (model(t_all, *res_full.x) - y_all) ** 2
    idx_keep  = np.argsort(residuos)[:len(t_all) - o]  # conservar los m-o menores

    # Paso 2: datos sin los o outliers
    t_sub = t_all[idx_keep]
    y_sub = y_all[idx_keep]

    # Paso 3: OLS sobre datos limpios
    t_start = time.perf_counter()
    res     = minimize(objetivo_ols, x0, args=(t_sub, y_sub),
                       method='L-BFGS-B', bounds=bounds)
    t_total = time.perf_counter() - t_start

    rows.append([o, f"{res.fun:.4f}", res.nit, res.nfev, f"{t_total:.4f}"])

headers = ["o", "f_ols(x*)", "#it", "#fcnt", "T(s)"]
print(tabulate(rows, headers=headers, tablefmt="grid"))
import numpy as np
from scipy.optimize import minimize
from tabulate import tabulate
import time

def model(t, x0, x1, x2, x3, x4):
    return x0 + (x1 * np.exp(-t * x3)) + (x2 * np.exp(-t * x4))

def objetivo_ols(x, t, y):
    return 0.5 * np.sum((model(t, *x) - y) ** 2)

# ─── Carga de datos ───────────────────────────────────────────────────────────
data  = np.loadtxt("txt/data_osborne1.txt")
t_all = data[:, 0]
y_all = data[:, 1]

x0     = [0.5, 1.5, -1.0, 0.01, 0.02]
bounds = [(0, 2), (0, 3), (-2, 0), (0, 0.5), (0, 0.5)]

# ─── Tabla OLS para o = 0..8 ──────────────────────────────────────────────────
# Para cada o:
#   1. Ajustar OLS sobre todos los datos → identificar los o residuos más grandes
#   2. Eliminar esos o puntos
#   3. Ajustar OLS sobre los m-o datos restantes → reportar f_ols(x*), #it, #fcnt, T(s)

header = ["o", "f_ols(x*)", "#it", "#fcnt", "T(s)"]
table  = []

for o in range(9):

    # Paso 1: OLS completo para detectar qué puntos son los o outliers
    res_full = minimize(objetivo_ols, x0, args=(t_all, y_all),
                        method='L-BFGS-B', bounds=bounds)
    residuos = (model(t_all, *res_full.x) - y_all) ** 2
    idx_keep = np.argsort(residuos)[:len(t_all) - o]   # conservar los m-o menores

    # Paso 2: datos sin los o outliers
    t_sub = t_all[idx_keep]
    y_sub = y_all[idx_keep]

    # Paso 3: OLS sobre datos limpios
    t_start = time.perf_counter()
    res     = minimize(objetivo_ols, x0, args=(t_sub, y_sub),
                       method='L-BFGS-B', bounds=bounds)
    t_total = time.perf_counter() - t_start

    table.append([o, f"{res.fun:.6f}", res.nit, res.nfev, f"{t_total:.5f}"])

print(tabulate(table, headers=header, tablefmt="grid"))

# Guardar solución del caso sin outliers (o=0, datos completos)
res_final = minimize(objetivo_ols, x0, args=(t_all, y_all),
                     method='L-BFGS-B', bounds=bounds)
#np.savetxt("txt/sol_osborne_ols.txt", res_final.x, fmt="%.6f")
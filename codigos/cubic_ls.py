import numpy as np
from scipy.optimize import minimize
import time

# ─────────────────────────────────────────────────────────────────────────────
#  Si tienes el data.txt, descomenta esta línea y comenta el bloque "data = ..."
# ─────────────────────────────────────────────────────────────────────────────
# data = np.loadtxt("txt/data.txt")

# Datos de la Tabla 4.1 (46 puntos, con 10 outliers en y=10.0)
data = np.array([
    [-1.0, -5.9907], [-0.9, -4.9602], [-0.8, -4.0419], [-0.7, -3.2048],
    [-0.6, -2.4872], [-0.5, -1.8734], [-0.4, 10.0000], [-0.3, 10.0000],
    [-0.2, 10.0000], [-0.1, 10.0000], [ 0.0, 10.0000], [ 0.1, 10.0000],
    [ 0.2, 10.0000], [ 0.3, 10.0000], [ 0.4, 10.0000], [ 0.5, 10.0000],
    [ 0.6,  0.3394], [ 0.7,  0.2647], [ 0.8,  0.1973], [ 0.9,  0.0937],
    [ 1.0, -0.0094], [ 1.1, -0.0932], [ 1.2, -0.1951], [ 1.3, -0.2705],
    [ 1.4, -0.3337], [ 1.5, -0.3820], [ 1.6, -0.3903], [ 1.7, -0.3647],
    [ 1.8, -0.2977], [ 1.9, -0.1713], [ 2.0,  0.0093], [ 2.1,  0.2223],
    [ 2.2,  0.5288], [ 2.3,  0.8963], [ 2.4,  1.3460], [ 2.5,  1.8668],
    [ 2.6,  2.4976], [ 2.7,  3.2084], [ 2.8,  4.0331], [ 2.9,  4.9619],
    [ 3.0,  5.9996], [ 3.1,  7.1581], [ 3.2,  8.4430], [ 3.3,  9.8757],
    [ 3.4, 11.4231], [ 3.5, 13.1256],
])

t = data[:, 0]
y = data[:, 1]

# ─────────────────────────────────────────────────────────────────────────────
#  Función de valor ordenado (OVO): (o+1)-ésimo residuo más grande
#  f_j(x) = (modelo(t_j, x) - y_j)^2
#  fovo_o(x) = el (o+1)-ésimo valor más grande de {f_j(x)}
# ─────────────────────────────────────────────────────────────────────────────
def fovo(x, o):
    residuals = (np.polyval(x[::-1], t) - y) ** 2
    return np.sort(residuals)[::-1][o]

# ─────────────────────────────────────────────────────────────────────────────
#  OLS: minimiza 0.5 * sum((modelo - y)^2)  — igual que tu código original
# ─────────────────────────────────────────────────────────────────────────────
def objetivo_ols(x):
    return 0.5 * np.sum((np.polyval(x[::-1], t) - y) ** 2)

x0     = [-1.0, -2.0, 1.0, -1.0]
bounds = [(-10, 10)] * 4

t_start = time.time()
res_ols = minimize(objetivo_ols, x0, method='L-BFGS-B', bounds=bounds)
t_ols   = time.time() - t_start

x_ols = res_ols.x

print(f"x*_ols = {x_ols}")
print(f"Tiempo total OLS: {t_ols:.4f} s\n")

# ─────────────────────────────────────────────────────────────────────────────
#  Tabla: f_ovo(x*_ols) para o = 0..15
#  OLS es un método directo (no iterativo en sentido OVO), por eso
#  #it y #fcnt se reportan con "—"
# ─────────────────────────────────────────────────────────────────────────────
print(f"{'o':>3} | {'f_ovo(x*_ols)':>14}")
print("-" * 22)
for o in range(16):
    fval = fovo(x_ols, o)
    print(f"{o:>3} | {fval:>14.4f}")


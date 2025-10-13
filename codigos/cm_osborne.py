import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

datos = np.loadtxt('data_osborne1.txt')
t = datos[:, 0]
y = datos[:, 1]

def osborne1(beta, x):
    return beta[0] + beta[1] * np.exp(-beta[3] * x) + beta[2] * np.exp(-beta[4] * x)

def objetivo(beta):
    y_modelo = osborne1(beta, t)
    return 0.5 * np.sum((y - y_modelo)**2)

beta0 = [0.5, 1.5, -1.0, 0.01, 0.02]
bounds = [
    (-2, 2),
    (-2, 2),
    (-2, 2),
    (0, 1),
    (0, 1)
]

res = minimize(objetivo, beta0, method='L-BFGS-B', bounds=bounds)

print("=" * 60)
print("AJUSTE FUNCIÓN DE OSBORNE 1 - L-BFGS-B (con ruido)")
print("=" * 60)
for i, b in enumerate(res.x, 1):
    print(f"β{i} = {b:.6f}")
print(f"\nValor de la función objetivo = {res.fun:.6e}")
print(f"Número de evaluaciones = {res.nfev}")
print(f"Optimización exitosa: {res.success}")
print(f"Mensaje: {res.message}")

y_fit = osborne1(res.x, t)
ss_res = np.sum((y - y_fit)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - ss_res/ss_tot
print(f"\nR² (ajuste) = {r2:.6f}")

os.makedirs("figuras", exist_ok=True)
tt = np.linspace(t.min(), t.max(), 400)
y_smooth = osborne1(res.x, tt)

plt.scatter(t, y, color='red', label='Datos con ruido', s=40)
plt.plot(tt, y_smooth, 'g--', lw=2, label='Ajuste modelo Osborne 1')
plt.savefig("figuras/osborne1_ruidoso.pdf", bbox_inches='tight')
plt.show()
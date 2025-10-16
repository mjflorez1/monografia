import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# Cargar datos
datos = np.loadtxt('data_gauss.txt')
t = datos[:, 0]
y = datos[:, 1]

def gauss(beta, x):
    return beta[0] * np.exp(-((x - beta[1])**2) / (2 * beta[2]**2))

def objetivo(beta):
    y_modelo = gauss(beta, t)
    return 0.5 * np.sum((y - y_modelo)**2)

beta0 = [0.4, 0.0, 2.0]

bounds = [
    (0, 1.0),
    (-5, 5),
    (0.1, 5.0)
]

res = minimize(objetivo, beta0, method='L-BFGS-B', bounds=bounds)

print("=" * 60)
print("AJUSTE FUNCIÓN GAUSSIANA - L-BFGS-B (con ruido)")
print("=" * 60)
print(f"β1 (amplitud) = {res.x[0]:.6f}")
print(f"β2 (media) = {res.x[1]:.6f}")
print(f"β3 (desv. est.) = {res.x[2]:.6f}")
print(f"\nValor de la función objetivo = {res.fun:.6e}")
print(f"Número de evaluaciones = {res.nfev}")
print(f"Optimización exitosa: {res.success}")
print(f"Mensaje: {res.message}")

y_fit = gauss(res.x, t)
ss_res = np.sum((y - y_fit)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - ss_res/ss_tot
print(f"\nR² (ajuste) = {r2:.6f}")

os.makedirs("figuras", exist_ok=True)
tt = np.linspace(t.min(), t.max(), 400)
y_smooth = gauss(res.x, tt)

plt.scatter(t, y, color='red', label='Datos con ruido', s=40, zorder=3)
plt.plot(tt, y_smooth, 'g', lw=2, label='Ajuste modelo Gaussiano', zorder=2)
plt.xlabel('t')
plt.ylabel('y')
plt.savefig("figuras/gauss_ols.pdf", bbox_inches='tight')
plt.show()
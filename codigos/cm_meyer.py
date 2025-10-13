import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

datos = np.loadtxt('data_meyer.txt')
t = datos[:, 0]
y = datos[:, 1]

def meyer(beta, x):
    return beta[0] * np.exp(beta[1] / (x + beta[2]))

def objetivo(beta):
    y_modelo = meyer(beta, t)
    return 0.5 * np.sum((y - y_modelo) ** 2)

beta0 = [2.5, 6000, 350]

bounds = [(0.01, 10), (1000, 10000), (100, 600)]
res_lbfgsb = minimize(objetivo, beta0, method='L-BFGS-B', bounds=bounds)

print("=" * 60)
print("AJUSTE FUNCIÓN DE MEYER - L-BFGS-B (con bounds)")
print("=" * 60)
print("\nCoeficientes estimados:")
print(f"β₁ = {res_lbfgsb.x[0]:.6f}")
print(f"β₂ = {res_lbfgsb.x[1]:.6f}")
print(f"β₃ = {res_lbfgsb.x[2]:.6f}")
print(f"\nValor de la función objetivo = {res_lbfgsb.fun:.6f}")
print(f"Número de iteraciones = {res_lbfgsb.nfev}")
print(f"Optimización exitosa: {res_lbfgsb.success}")

y_fit = meyer(res_lbfgsb.x, t)

ss_res = np.sum((y - y_fit)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - (ss_res / ss_tot)
print(f"\nR² (mejor ajuste) = {r2:.6f}")

xx = np.linspace(t.min() - 5, t.max() + 5, 400)
y_fit_xx = meyer(res_lbfgsb.x, xx)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))

ax1.scatter(t, y, label='Datos observados', color='red', s=50, zorder=3)
ax1.plot(xx, y_fit_xx, 'g--', linewidth=2, label='Ajuste Meyer')

ax2.scatter(t, y, color='red', s=50, zorder=3)
ax2.plot(xx, y_fit_xx, 'g--', linewidth=2)
ax2.set_xlim(60, 90)
ax2.set_ylim(9000, 22000)

plt.tight_layout()
plt.savefig("figuras/meyer_ls.pdf", bbox_inches = 'tight')
plt.show()
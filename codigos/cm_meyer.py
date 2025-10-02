import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

data = np.loadtxt("data_meyer.txt")
t = data[:, 0]
y = data[:, 1]

# Función de Meyer: f(x, beta) = beta[0] * exp(beta[1] / (x + beta[2]))
def meyer(beta, x):
    return beta[0] * np.exp(beta[1] / (x + beta[2]))

# Función de residuos
def residuals(beta, x, y):
    return y - meyer(beta, x)

# Valores iniciales para los parámetros
beta_0 = np.array([1.0, 1.0, 1.0])

# Ajuste usando mínimos cuadrados no lineales
result = least_squares(residuals, beta_0, args=(t, y))
beta_hat = result.x
y_hat = meyer(beta_hat, t)
resid = y - y_hat

# Cálculo de la matriz Jacobiana en el óptimo
J = result.jac
n, p = len(t), len(beta_hat)

# Estimación de la varianza residual
sigma2_hat = (resid @ resid) / (n - p)

# Matriz de covarianza de los parámetros
# cov(beta) ≈ sigma² * (J'J)^(-1)
cov_beta = sigma2_hat * np.linalg.inv(J.T @ J)
stderr_beta = np.sqrt(np.diag(cov_beta))

# R² - coeficiente de determinación
ss_res = np.sum(resid**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - (ss_res / ss_tot)

# Resultados numéricos
print("=" * 60)
print("AJUSTE FUNCIÓN DE MEYER - MÍNIMOS CUADRADOS NO LINEALES")
print("=" * 60)
print(f"\nCoeficientes estimados:")
print(f"β₁ = {beta_hat[0]:.6f} ± {stderr_beta[0]:.6f}")
print(f"β₂ = {beta_hat[1]:.6f} ± {stderr_beta[1]:.6f}")
print(f"β₃ = {beta_hat[2]:.6f} ± {stderr_beta[2]:.6f}")
print(f"\nVarianza residual estimada = {sigma2_hat:.6f}")
print(f"R² = {r2:.6f}")
print(f"Número de iteraciones = {result.nfev}")
print("=" * 60)

# Graficar
xx = np.linspace(t.min() - 5, t.max() + 5, 400)
y_hat_xx = meyer(beta_hat, xx)

plt.figure(figsize=(10, 6))
plt.scatter(t, y, label='datos observados', marker='o', color='r', s=50, zorder=3)
plt.plot(xx, y_hat_xx, label='ajuste Meyer', linewidth=2, color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ajuste de la Función de Meyer')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
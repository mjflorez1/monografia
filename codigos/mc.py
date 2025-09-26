import numpy as np
import matplotlib.pyplot as plt

# Cargar datos desde el txt
data = np.loadtxt("data.txt")
t = data[:, 0]
y = data[:, 1]

# Construir matriz de diseño
X = np.vstack([np.ones_like(t), t, t**2, t**3]).T

# Ajuste OLS
beta_ols, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
y_hat_ols = X @ beta_ols
resid_ols = y - y_hat_ols

# Estimación de la varianza y errores estándar
n, p = X.shape
sigma2_hat = (resid_ols @ resid_ols) / (n - p)
cov_beta = sigma2_hat * np.linalg.inv(X.T @ X)
stderr_beta = np.sqrt(np.diag(cov_beta))

# Resultados numéricos
print("Coeficientes OLS =", beta_ols)
print("Errores estándar =", stderr_beta)
print("Varianza residual estimada =", sigma2_hat)

# Graficar
tt = np.linspace(t.min()-0.2, t.max()+0.2, 400)
Xtt = np.vstack([np.ones_like(tt), tt, tt**2, tt**3]).T
y_ols_tt = Xtt @ beta_ols

plt.figure(figsize=(8,5))
plt.scatter(t, y, label='datos (con outliers)', marker='o',color='r')
plt.plot(tt, y_ols_tt, label='ajuste OLS', linewidth=1.5)
plt.xlabel('t')
plt.ylabel('y')
plt.show()
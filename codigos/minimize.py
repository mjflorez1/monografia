import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Cargar los datos
datos = np.loadtxt('data.txt', skiprows=1)
t = datos[:, 0]
y = datos[:, 1]

# Modelo verdadero (sin ruido)
x_star = [0, 2, -3, 1]
w = x_star[0] + x_star[1]*t + x_star[2]*t**2 + x_star[3]*t**3

# Función objetivo: suma de cuadrados de los errores
def objetivo(x):
    y_modelo = np.polyval(x[::-1], t)
    return np.sum((y - y_modelo) ** 2)

# Condición inicial y límites
x0 = [-1, -2, 1, -1]
bounds = [(-10, 10)] * 4  # Límite para cada coeficiente

# Resolver usando minimize (usamos método 'L-BFGS-B' por permitir bounds)
res = minimize(objetivo, x0, method='L-BFGS-B', bounds=bounds)

print("Coeficientes del polinomio ajustado con minimize:")
print(res.x)

# Evaluar el modelo ajustado
y_fit = np.polyval(res.x[::-1], t)

# Graficar
plt.figure()
plt.scatter(t, y, label='Outliers', color='blue')
plt.plot(t, w, 'r', linewidth=1.5, label='Modelo verdadero')
plt.plot(t, y_fit, 'g--', linewidth=1.5, label='Ajuste con minimize')
plt.savefig("figuras/OLS.pdf", bbox_inches= 'tight')
plt.show()
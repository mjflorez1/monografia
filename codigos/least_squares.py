import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Cargar los datos desde el archivo, ignorando la primera línea
datos = np.loadtxt('data.txt', skiprows=1)
t = datos[:, 0]
y = datos[:, 1]

# Modelo verdadero sin ruido
x_star = [0, 2, -3, 1]
w = x_star[0] + x_star[1]*t + x_star[2]*t**2 + x_star[3]*t**3

# Función de residuos para least_squares
def residuals(x):
    return np.polyval(x[::-1], t) - y

# Condiciones iniciales y límites
x0 = [-1, -2, 1, -1]
bounds = (-10 * np.ones(4), 10 * np.ones(4))

# Resolver con least_squares (equivalente a lsqnonlin)
res = least_squares(residuals, x0, bounds=bounds, verbose=2)

print("Coeficientes del polinomio ajustado con least_squares:")
print(res.x)

# Evaluar el modelo ajustado
y_fit = np.polyval(res.x[::-1], t)

# Graficar resultados
plt.figure()
plt.scatter(t, y, label='Polinomio con outliers', color='blue')
plt.plot(t, w, 'r', linewidth=1.5, label='Modelo verdadero')
plt.plot(t, y_fit, 'g-.', linewidth=1.5, label='Ajuste con least_squares')
plt.legend()
plt.show()
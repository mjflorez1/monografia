import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo, saltando la primera línea
datos = np.loadtxt('data.txt', skiprows=1)

# Extraer columnas
t = datos[:, 0]
y = datos[:, 1]

# Coeficientes del modelo verdadero
x_star = [0, 2, -3, 1]

# Modelo verdadero (sin ruido)
w = x_star[0] + x_star[1] * t + x_star[2] * t**2 + x_star[3] * t**3

# Ajuste polinómico de grado 3
coeff_polyfit = np.polyfit(t, y, 3)

# Evaluar el polinomio ajustado
y_polyfit = np.polyval(coeff_polyfit, t)

# Mostrar coeficientes ajustados
print("Coeficientes del polinomio ajustado con polyfit:")
print(coeff_polyfit)

# Gráfica comparativa
plt.figure()
plt.scatter(t, y, label='Outliers', color='blue')
plt.plot(t, w, 'r', linewidth=1.5, label='Modelo verdadero')
plt.plot(t, y_polyfit, 'm--', linewidth=1.5, label='Ajuste con polyfit')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Comparación del ajuste POLYFIT con los datos generados')
plt.legend()
plt.show()
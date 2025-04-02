import numpy as np
import random

# Definimos el modelo cúbico
def model(t, x1, x2, x3, x4):
    return x1 + x2 * t + x3 * t**2 + x4 * t**3

def f_i(ti, yi, x):
    return (model(ti, *x) - yi) ** 2

# Leer datos desde el archivo
data = np.loadtxt("data.txt", skiprows=1)
t = data[:, 0]
y = data[:, 1]

# Punto inicial
xk = np.array([-1, -2, 1, -1])

# Calcular valores de la función auxiliar
m = len(t)
faux = np.zeros(m)
for i in range(m):
    faux[i] = f_i(t[i], y[i], xk)

# Índices de faux organizados
fsort = np.sort(faux)
indices = np.argsort(faux)

# Posición p y epsilon
p = 36
eps = 1
Ieps = []

# Valor de la función objetivo
fovo = fsort[p]

# Construcción del conjunto I(x,eps)
for i in range(m):
    if np.abs(fovo - fsort[i]) < eps:
        Ieps.append(indices[i])

print(Ieps)
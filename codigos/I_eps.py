import numpy as np
import random

#Definimos el modelo cubico
def model(t, x1, x2, x3, x4):
  return x1 + (x2 * t) + (x3 * t ** 2) + (x4 * t ** 3)

def f_i(ti, yi, x):
  return (model(ti, *x) - yi) ** 2

# Numero de datos
m = 46

# Definimos los valores de ti (de 1 a 46)
t = np.linspace(-1, 3.5, m)

# "Solución exacta"
xstar = np.array([0, 2, -3, 1])

# Generando datos
y = 10 * np.ones(m)
random.seed(1234)
noise = 0.01

for i in range(6):
  y[i] = model(t[i], *xstar) + random.uniform(-noise, noise)

for i in range(16, m):
  y[i] = model(t[i], *xstar) + random.uniform(-noise, noise)

# Punto inicial
xk = np.array([-1,-2,1,-1])

# Funcion auxiliar para ordenar
faux = np.zeros(m)

for i in range(m):
  faux[i] = f_i(t[i], y[i], xk)

#Indices de y y faux
original_index = np.arange(1, m)
y_index = np.arange(1, m)

#Indices de faux organizados
fsort = np.sort(faux)
indices = np.argsort(faux)

#Posicion p y epsilon
p = 36
eps = 1
Ieps = []

# Valor de la funcion ovo
fovo = fsort[p]

# Construccion del conjunto I(x,eps)
for i in range(m):
    if np.abs(fovo - fsort[i]) < eps:
        Ieps.append(indices[i])

print(Ieps)
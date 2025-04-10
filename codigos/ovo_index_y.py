import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

#Definimos el modelo cubico
def model(t, x1, x2, x3, x4):
  return x1 + (x2 * t) + (x3 * t ** 2) + (x4 * t ** 3)

def f_i(ti, yi, x):
  return (model(ti, *x) - yi) ** 2

#Definimos los valores de ti (de 1 a 46)
m = 47
t = np.linspace(-1, 3.5, m)
xstar = np.array([0, 2, -3, 1])
y = -10 * np.ones(m)
random.seed(1234)

for i in range(6):
  y[i] = model(t[i], *xstar) + random.uniform(-0.01, 0.01)

for i in range(16, m):
  y[i] = model(t[i], *xstar) + random.uniform(-0.01, 0.01)

xk = np.ones(4)
faux = np.zeros(m)

for i in range(m):
  faux[i] = f_i(t[i], y[i], xk)

#Indices de y y faux
original_index = np.arange(1, m)
y_index = np.arange(1, m)

#Indices de faux organizados
faux_sort = np.sort(faux)
sorted_index = np.argsort(faux) + 1

#Posicion p y epsilon
p = 35
eps = 1

#valor f(p) para faux y y
fovo = faux[p - 1]
y_p = y[p - 1]

#calculamos el intervalo I_eps
l_bound = y_p - eps
u_bound = y_p + eps

idx = []
for i in range(1, m - 1):
  if l_bound <= y[i] <= u_bound:
    idx.append(i + 1)

#print(idx)

#print(original_index)
#print(faux)
#print(fovo)
print(y_p)
#print(faux_sort)
#print(sorted_index)

print(y)
#print(y_index)
print(f'Valores de faux dentro de I_eps: ', idx)

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Definimos el modelo cubico
def model(t,x1,x2,x3,x4):
    return x1 + x2 * t + x3 * t**2 + x4 * t**3

def f_i(ti,yi,x):
    return (model(ti,*x) - yi)**2

#Definimos los valores de t_i (de 1 a 46)
m = 47
t = np.linspace(-1,3.5,m)
xstar = np.array([0, 2, -3, 1])

y = 10 * np.ones(m)
random.seed(1234)

for i in range(6):
    y[i] = model(t[i],*xstar) + random.uniform(-0.01, 0.01)

for i in range(16,m):
    y[i] = model(t[i],*xstar) + random.uniform(-0.01, 0.01)

xk = np.ones(4)
faux = np.zeros(m)

for i in range(m):
    faux[i] = f_i(t[i],y[i],xk)

faux = np.sort(faux)

p = 40

fovo = faux[p-1]

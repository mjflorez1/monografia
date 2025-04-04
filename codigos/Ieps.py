import numpy as np

# Modelo cúbico
def model(t, x1, x2, x3, x4):
    return x1 + (x2 * t) + (x3 * t ** 2) + (x4 * t ** 3)

def f_i(ti, yi, x):
    return (model(ti, *x) - yi) ** 2

# Leer datos desde archivo
with open("data.txt", "r") as f:
    lines = f.readlines()

m = int(lines[0].strip())

t = []
y = []

for line in lines[1:]:
    ti, yi = map(float, line.strip().split())
    t.append(ti)
    y.append(yi)

t = np.array(t)
y = np.array(y)

# Solución exacta
xstar = np.array([0, 2, -3, 1])

# Punto inicial
xk = np.array([-1, -2, 1, -1])

# Evaluar función para cada punto
faux = np.zeros(m)
for i in range(m):
    faux[i] = f_i(t[i], y[i], xk)

# Ordenar resultados
original_index = np.arange(m)  # Índices de 0 a m-1
fsort = np.sort(faux)
indices = np.argsort(faux)

# Parámetros
p = 36
eps = 0.8
Ieps = []

# Valor de referencia (ovo)
fovo = fsort[p]


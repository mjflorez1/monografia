import numpy as np

# Modelo cúbico
def model(t, x1, x2, x3, x4):
    return x1 + (x2 * t) + (x3 * t ** 2) + (x4 * t ** 3)

# Función de error
def f_i(ti, yi, x):
    return (model(ti, *x) - yi) ** 2

# Leer datos desde data.txt
with open("data.txt", "r") as f:
    lines = f.readlines()

m = int(lines[0].strip())  # Número de datos

# Inicializamos listas para t y y
t = []
y = []
for line in lines[1:]:
    ti, yi = map(float, line.strip().split())
    t.append(ti)
    y.append(yi)

t = np.array(t)
y = np.array(y)

# Definimos el punto en el que evaluamos
x = np.array([-1, -2, 1, -1])

# Calculamos f_i(x) para cada dato
f_values = np.array([f_i(t[i], y[i], x) for i in range(m)])

# Parámetros
p = 35
eps = 10

# Ordenamos los valores y obtenemos índices ordenados
sorted_idx = np.argsort(f_values)
sorted_f = f_values[sorted_idx]

# f(x) es el p-ésimo valor ordenado
f_p = sorted_f[p]

# Construcción de I_eps con np.abs() para incluir todos los valores cercanos
I_eps = [int(sorted_idx[i]) for i in range(m) if np.abs(sorted_f[i] - f_p) <= eps]

print("f(x) (p-ésimo valor) =", f_p)
print("I_{eps}(x) =", I_eps)
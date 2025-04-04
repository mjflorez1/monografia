import numpy as np

# Modelo cúbico
def model(t, x1, x2, x3, x4):
    return x1 + x2 * t + x3 * t**2 + x4 * t**3

# Función de error para cada dato
def f_i(ti, yi, x):
    return (model(ti, *x) - yi)**2

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

# Definimos el punto en el que evaluamos (por ejemplo, el punto inicial)
x = np.array([-1, -2, 1, -1])

# Calculamos f_i(x) para cada dato
f_values = np.array([f_i(t[i], y[i], x) for i in range(m)])

# Parámetros de la definición del intervalo:
p = 36           # p-ésimo valor (puedes ajustar este parámetro)
eps = 100        # ε (puedes cambiarlo, por ejemplo, 0.8, 1.0, etc.)

# Ordenamos los f_values y obtenemos los índices de ordenamiento
sorted_indices = np.argsort(f_values)
sorted_f = f_values[sorted_indices]

# f(x) se define como el p-ésimo valor (recordar que los índices en Python empiezan en 0)
f_p = sorted_f[p]

# Construimos I₍ε₎(x): índices i tales que f_i(x) ∈ [f_p, f_p + eps]
I_eps = [int(sorted_indices[i]) for i in range(m) if sorted_f[i] >= f_p and sorted_f[i] <= f_p + eps]

print("f(x) (p-ésimo valor) =", f_p)
print("I₍ε₎(x) =", I_eps)
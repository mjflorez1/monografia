# Bibliotecas esenciales
import numpy as np
import pandas as pd
import time
from scipy.optimize import linprog

# Definición del modelo cúbico
def model(t, x1, x2, x3, x4):
    return x1 + (x2 * t) + x3 * (t**2) + x4 * (t**3)

# Funciones de error cuadrático
def f_i(t_i, y_i, x):
    return 0.5 * ((model(t_i, *x) - y_i) ** 2)

# Gradientes de las funciones de error
def grad_f_i(t_i, y_i, x, grad):
    diff = model(t_i, *x) - y_i
    grad[0] = diff * 1
    grad[1] = diff * t_i
    grad[2] = diff * t_i**2
    grad[3] = diff * t_i**3
    return grad[:]

# Montamos el conjunto de índices I_delta
def mount_Idelta(fovo, faux, indices, delta, Idelta):
    k = 0
    for i in range(len(faux)):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

def ovo_algorithm(t, y, o):
    # Parámetros algorítmicos
    epsilon = 1e-8
    delta   = 1e-3
    deltax  = 1.0
    theta   = 0.5
    n = 5
    q = o  # usamos "o" como índice del valor ordenado
    max_iter = 1000
    max_iter_armijo = 100

    # Solución inicial
    xk = np.array([-1, -2, 1, -1])

    # Definimos arrays necesarios
    faux    = np.zeros(m)
    Idelta  = np.zeros(m, dtype=int)
    c = np.zeros(n)
    c[-1] = 1

    iter = 1
    fcnt = 0

    start_time = time.time()

    while iter <= max_iter:
        iter_armijo = 0

        # Restricciones de caja
        x0_bounds = (max(-10 - xk[0], -deltax), min(10 - xk[0], deltax))
        x1_bounds = (max(-10 - xk[1], -deltax), min(10 - xk[1], deltax))
        x2_bounds = (max(-10 - xk[2], -deltax), min(10 - xk[2], deltax))
        x3_bounds = (max(-10 - xk[3], -deltax), min(10 - xk[3], deltax))
        x4_bounds = (None, 0)

        # Cálculo de las funciones de error
        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)
        fcnt += m

        # Ordenamos funciones de error
        indices = np.argsort(faux)
        faux = np.sort(faux)
        fxk = faux[q]

        # Computamos Idelta
        nconst = mount_Idelta(fxk, faux, indices, delta, Idelta)

        # Montamos restricciones
        A = np.zeros((nconst, n))
        b = np.zeros(nconst)
        grad = np.zeros((nconst, n - 1))

        for i in range(nconst):
            ind = Idelta[i]
            grad_f_i(t[ind], y[ind], xk, grad[i, :])
            A[i, :-1] = grad[i, :]
            A[i, -1] = -1

        res = linprog(c, A_ub=A, b_ub=b,
                      bounds=[x0_bounds, x1_bounds, x2_bounds, x3_bounds, x4_bounds],
                      method="highs")

        dk = res.x
        mkd = dk[-1]

        if abs(mkd) < epsilon:
            break

        # Armijo
        alpha = 1
        xktrial = xk.copy()

        while iter_armijo <= max_iter_armijo:
            iter_armijo += 1
            xktrial = xk + (alpha * dk[:-1])

            for i in range(m):
                faux[i] = f_i(t[i], y[i], xktrial)
            fcnt += m

            faux = np.sort(faux)
            fxktrial = faux[q]

            if fxktrial < fxk + (theta * alpha * mkd):
                fxk = fxktrial
                break

            alpha *= 0.5

        xk = xktrial
        iter += 1

    elapsed = time.time() - start_time
    return [o, fxk, iter, fcnt, elapsed]

# ================================
# Ejecución para outliers o=0,...,12
# ================================
data = np.loadtxt("data.txt")
t = data[:, 0]
y = data[:, 1]
m = len(t)

results = []
for o in range(13):  # de 0 a 12
    results.append(ovo_algorithm(t, y, o))

df = pd.DataFrame(results, columns=["o", "f(x*)", "#it", "#fcnt", "Time"])
print(df)
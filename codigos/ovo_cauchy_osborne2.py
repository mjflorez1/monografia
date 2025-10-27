# Bibliotecas esenciales
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

def model(t, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    res = (x1 * np.exp(-t * x5) + x2 * np.exp(-x6 * (t - x9) ** 2) + x3 * np.exp(-x7 * (t - x10) ** 2)
        + x4 * np.exp(-x8 * (t - x11) ** 2))
    return res

def f_i(t_i, y_i, x):
    return 0.5 * ((model(t_i, *x) - y_i) ** 2)

def grad_f_i(t_i, y_i, x, grad):
    diff = model(t_i, *x) - y_i
    t = t_i
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = x

    grad[0]  = diff * np.exp(-t * x5)
    grad[1]  = diff * np.exp(-x6 * (t - x9) ** 2)
    grad[2]  = diff * np.exp(-x7 * (t - x10) ** 2)
    grad[3]  = diff * np.exp(-x8 * (t - x11) ** 2)
    grad[4]  = diff * (-t) * x1 * np.exp(-t * x5)
    grad[5]  = diff * (-((t - x9) ** 2)) * x2 * np.exp(-x6 * (t - x9) ** 2)
    grad[6]  = diff * (-((t - x10) ** 2)) * x3 * np.exp(-x7 * (t - x10) ** 2)
    grad[7]  = diff * (-((t - x11) ** 2)) * x4 * np.exp(-x8 * (t - x11) ** 2)
    grad[8]  = diff * (2 * x2 * x6 * (t - x9)) * np.exp(-x6 * (t - x9) ** 2)
    grad[9]  = diff * (2 * x3 * x7 * (t - x10)) * np.exp(-x7 * (t - x10) ** 2)
    grad[10] = diff * (2 * x4 * x8 * (t - x11)) * np.exp(-x8 * (t - x11) ** 2)
    return grad[:]

def mount_Idelta(fovo, faux, indices, delta, Idelta):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

def ovo_algorithm(t, y):
    epsilon = 1e-8
    delta   = 1e-3
    deltax  = 0.1
    theta   = 0.5
    n = 12
    q = 64
    max_iter = 1000
    max_iter_armijo = 100
    iter = 1

    xk = np.array([1.3, 0.65, 0.65, 0.7, 0.6, 3, 5, 7, 2, 4.5, 5.5])

    xktrial = np.zeros(n - 1)
    faux    = np.zeros(m)
    Idelta  = np.zeros(m, dtype=int)

    c = np.zeros(n)
    c[-1] = 1

    while iter <= max_iter:
        iter_armijo = 0


        bounds = []
        for i in range(len(xk)):
            bounds.append((-deltax, deltax))
        bounds.append((None, 0))

        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)

        indices = np.argsort(faux)
        faux = np.sort(faux)

        fxk = faux[q]

        nconst = mount_Idelta(fxk, faux, indices, delta, Idelta)

        A = np.zeros((nconst, n))
        b = np.zeros(nconst)
        grad = np.zeros((nconst, n - 1))

        for i in range(nconst):
            ind = Idelta[i]
            grad_f_i(t[ind], y[ind], xk, grad[i, :])
            A[i, :-1] = grad[i, :]
            A[i, -1] = -1

        # Resolver subproblema convexo
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)

        dk = res.x
        mkd = dk[-1]

        # Criterio de parada
        if abs(mkd) < epsilon:
            break

        # Búsqueda lineal tipo Armijo
        alpha = 1
        while iter_armijo <= max_iter_armijo:
            iter_armijo += 1
            xktrial = xk + alpha * dk[:-1]
            for i in range(m):
                faux[i] = f_i(t[i], y[i], xktrial)
            faux = np.sort(faux)
            fxktrial = faux[q]
            if fxktrial < fxk + theta * alpha * mkd:
                break
            alpha *= 0.5

        print(iter, fxk, mkd, iter_armijo)

        xk = xktrial
        iter += 1
        np.savetxt("txt/sol_osborne2_cauchy.txt", xk, fmt="%.6f")

    print("Solución final:", xk)
    return xk

# Cargar datos y ejecutar algoritmo
data = np.loadtxt("txt/data_osborne2.txt")
t = data[:, 0]
y = data[:, 1]
m = len(t)

xk_final = ovo_algorithm(t, y)
y_pred = model(t, *xk_final)

plt.scatter(t, y, label="Datos")
plt.plot(t, y_pred, label="Ajuste")
plt.legend()
#plt.savefig("figuras/osborne2_cauchy.pdf", bbox_inches="tight")
plt.show()
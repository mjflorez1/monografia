import numpy as np
from scipy.optimize import linprog, minimize
import matplotlib.pyplot as plt
from tabulate import tabulate
import os

# ==============================
# Modelo de Osborne 1
# ==============================
def model(t, x0, x1, x2, x3, x4):
    return x0 + (x1 * np.exp(-t * x3)) + (x2 * np.exp(-t * x4))

def f_i(t_i, y_i, x):
    return 0.5 * ((model(t_i, *x) - y_i) ** 2)

def grad_f_i(t_i, y_i, x, grad):
    diff = model(t_i, *x) - y_i
    grad[0] = diff
    grad[1] = diff * np.exp(-t_i * x[3])
    grad[2] = diff * np.exp(-t_i * x[4])
    grad[3] = diff * -t_i * x[1] * np.exp(-t_i * x[3])
    grad[4] = diff * -t_i * x[2] * np.exp(-t_i * x[4])
    return grad

def mount_Idelta(fovo, faux, indices, epsilon, Idelta):
    k = 0
    for i in range(len(faux)):
        if abs(fovo - faux[i]) <= epsilon:
            Idelta[k] = indices[i]
            k += 1
    return k

# ==============================
# Algoritmo OVO
# ==============================
def ovo(t, y):
    stop = 1e-8
    epsilon = 3e-3
    delta = 1.7
    theta = 0.5
    n = 6
    m = len(t)
    q = 27

    max_iter = 200
    max_iter_armijo = 35
    iter = 1

    xk = np.array([0.5, 1.5, -1.0, 0.01, 0.02])
    xktrial = np.zeros(5)
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)
    
    header = ["f(xk)", "Iter", "IterArmijo", "Mk(d)", "ncons", "Idelta"]
    table = []

    c = np.zeros(n)
    c[-1] = 1

    while iter <= max_iter:    
        iter_armijo = 0

        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)

        indices = np.argsort(faux)
        faux_sorted = faux[indices]
        fxk = faux_sorted[q]
        nconst = mount_Idelta(fxk, faux, indices, epsilon, Idelta)
        
        A = np.zeros((nconst, n))
        b = np.zeros(nconst)

        for i in range(nconst):
            ind = Idelta[i]
            grad = np.zeros(5)
            grad_f_i(t[ind], y[ind], xk, grad)
            A[i, :-1] = grad
            A[i, -1] = -1

        res = linprog(
            c, A_ub=A, b_ub=b, 
            bounds=[(-delta, delta), (-delta, delta), (-delta, delta), (-delta, delta), (-delta, delta), (None, 0)]
        )
        
        dk = res.x
        mkd = dk[-1]
        if abs(mkd) < stop:
            break

        alpha = 1
        while iter_armijo <= max_iter_armijo:
            iter_armijo += 1
            xktrial = xk + (alpha * dk[:-1])
            faux_trial = np.array([f_i(t[i], y[i], xktrial) for i in range(m)])
            fxktrial = np.sort(faux_trial)[q]
            if fxktrial <= fxk + (theta * alpha * mkd):
                break
            alpha *= 0.5

        table.append([fxk, iter, iter_armijo, mkd, nconst, Idelta[:min(5, nconst)].tolist()])
        xk = xktrial
        iter += 1
    
    print("\n" + "=" * 60)
    print("RESULTADOS MÉTODO OVO")
    print("=" * 60)
    print(tabulate(table, headers=header, tablefmt="grid"))
    print("Solución final:", xk)
    return xk

# ==============================
# Ajuste clásico L-BFGS-B
# ==============================
def ajuste_lbfgsb(t, y):
    def objetivo(beta):
        return 0.5 * np.sum((y - model(t, *beta)) ** 2)

    beta0 = [0.5, 1.5, -1.0, 0.01, 0.02]
    bounds = [(-2, 2), (-2, 2), (-2, 2), (0, 1), (0, 1)]
    res = minimize(objetivo, beta0, method='L-BFGS-B', bounds=bounds)

    print("\n" + "=" * 60)
    print("AJUSTE CLÁSICO L-BFGS-B")
    print("=" * 60)
    for i, b in enumerate(res.x, 1):
        print(f"β{i} = {b:.6f}")
    print(f"\nValor función objetivo = {res.fun:.6e}")
    print(f"Éxito optimización: {res.success}")
    print(f"Mensaje: {res.message}")

    return res.x

# ==============================
# Ejecución y gráfica comparativa
# ==============================
os.makedirs("figuras", exist_ok=True)
data = np.loadtxt("data_osborne1.txt")
t = data[:, 0]
y = data[:, 1]

# OVO
x_ovo = ovo(t, y)
y_ovo = model(t, *x_ovo)

# L-BFGS-B
x_lbfgsb = ajuste_lbfgsb(t, y)
tt = np.linspace(t.min(), t.max(), 400)
y_lbfgsb = model(tt, *x_lbfgsb)

# ==============================
# GRAFICAR AMBOS AJUSTES
# ==============================
plt.figure()
plt.scatter(t, y, color="gray", s=40, alpha=0.7, label="$Observaciones$")
plt.plot(tt, y_lbfgsb, "m-", lw=2, label="$OLS$")
plt.plot(t, y_ovo, "y-", lw=2, label="$OVO$")
plt.legend(loc="upper right")
plt.savefig("figuras/comparacion_osborne.pdf", bbox_inches="tight")
plt.show()
import numpy as np
from scipy.optimize import linprog
from tabulate import tabulate
import matplotlib.pyplot as plt

def model(t_i, x0, x1, x2):
    return x0 * np.exp(x1 / (t_i + x2))

def f_i(t_i, y_i, x):
    return 0.5 * ((model(t_i,*x) - y_i) ** 2)

def grad_f_i(t_i, y_i, x, grad):
    x0, x1, x2 = x
    z = np.exp(x1 / (t_i + x2))
    diff = (x0 * z) - y_i
    grad[0] = diff * z
    grad[1] = diff * x0 * z / (t_i + x2)
    grad[2] = diff * (-x0 * x1 * z / ((t_i + x2)**2))
    return grad

def mount_Idelta(fovo, faux, indices, epsilon, Idelta, m):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= epsilon:
            Idelta[k] = indices[i]
            k += 1
    return k

def ovo(t, y):
    stop = 2e+0
    epsilon = 1e+8
    delta = 1e+1
    theta = 0.3
    n = 4
    m = len(t)
    q = 12
    max_iter = 100
    max_iter_armijo = 30
    iter = 1

    xk = np.array([0.02, 4000.0, 250.0])
    xktrial = np.zeros(3)
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)

    header = ["f(xk)", "Iter", "IterArmijo", "Mk(d)", "ncons", "Idelta"]
    table = []

    c = np.zeros(n)
    c[-1] = 1

    while iter <= max_iter:
        iter_armijo = 0

        x0_bounds = (-delta, delta)
        x1_bounds = (-delta, delta)
        x2_bounds = (-delta, delta)
        x3_bounds = (None, 0)

        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)

        indices = np.argsort(faux)
        faux_sorted = faux[indices]
        fxk = faux_sorted[q]

        nconst = mount_Idelta(fxk, faux, indices, epsilon, Idelta, m)

        A = np.zeros((nconst, n))
        b = np.zeros(nconst)

        for i in range(nconst):
            ind = Idelta[i]
            grad = np.zeros(3)
            grad_f_i(t[ind], y[ind], xk, grad)
            A[i, :-1] = grad
            A[i, -1] = -1

        res = linprog(c, A_ub=A, b_ub=b, 
                     bounds=[x0_bounds, x1_bounds, x2_bounds, x3_bounds],
                     method='highs')
        
        dk = res.x
        mkd = dk[-1]

        if abs(mkd) < stop:
            break

        alpha = 1
        while iter_armijo <= max_iter_armijo:
            iter_armijo += 1
            xktrial = xk + (alpha * dk[:-1])
            
            faux_trial = np.zeros(m)
            for i in range(m):
                faux_trial[i] = f_i(t[i], y[i], xktrial)
            
            faux_trial_sorted = np.sort(faux_trial)
            fxktrial = faux_trial_sorted[q]
            
            if fxktrial <= fxk + (theta * alpha * mkd):
                break
            alpha *= 0.5

        table.append([fxk, iter, iter_armijo, mkd, nconst, Idelta[:min(5, nconst)].tolist()])
        xk = xktrial
        iter += 1

    print(tabulate(table, headers=header, tablefmt="grid"))
    print('Solución final:', xk)
    return xk

data = np.loadtxt('data_meyer.txt')
t = data[:, 0]
y = data[:, 1]

xk_final = ovo(t, y)
y_pred = model(t, *xk_final)

plt.scatter(t, y, color="blue", alpha=0.6, label="Datos observados")
plt.plot(t, y_pred, color="red", linewidth=2, label="Modelo ajustado OVO")
plt.savefig("figuras/ovo_meyer_cauchy.pdf", bbox_inches="tight", dpi=150)
plt.show()
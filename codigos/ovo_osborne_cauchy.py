import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from tabulate import tabulate
import time

def model(t, x0, x1, x2, x3, x4):
    return x0 + (x1 * np.exp(-t * x3)) + (x2 * np.exp(-t * x4))

def f_i(t_i, y_i, x):
    return 0.5 * ((model(t_i,*x) - y_i) ** 2)

def grad_f_i(t_i, y_i, x, grad):
    diff = model(t_i, *x) - y_i
    grad[0] = diff * 1
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

def ovo(t, y):
    stop = 1e-8
    epsilon = 3e-3
    delta = 1.7
    theta = 0.5
    n = 6
    m = len(t)
    q = 24
    max_iter = 200
    max_iter_armijo = 35
    iter = 1

    xk = np.array([0.5, 1.5, -1.0, 0.01, 0.02])
    xktrial = np.zeros(5)
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)
    
    header = ["f(xk)", "Iter", "IterArmijo", "Mk(d)", "ncons", "Idelta, Tiempo (s)"]
    table = []

    c = np.zeros(n)
    c[-1] = 1

    start_time = time.time()
    while iter <= max_iter:    
        iter_armijo = 0
        
        x0_bounds = (-delta, delta)
        x1_bounds = (-delta, delta)
        x2_bounds = (-delta, delta)
        x3_bounds = (-delta, delta)
        x4_bounds = (-delta, delta)
        x5_bounds = (None, 0)

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

        res = linprog(c, A_ub=A, b_ub=b, 
                     bounds=[x0_bounds, x1_bounds, x2_bounds, x3_bounds, x4_bounds, x5_bounds])
        
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

        xk = xktrial
        iter += 1
        elapsed = time.time() - start_time
        table.append([fxk, iter, iter_armijo, mkd, nconst, Idelta[:min(5, nconst)].tolist(),elapsed])
        np.savetxt('txt/sol_osborne_cauchy.txt',xk,fmt='%.6f')
    
    print(tabulate(table, headers=header, tablefmt="grid"))
    print('Solución final:', xk)
    print(fxk)
    return xk

data = np.loadtxt("txt/data_osborne1.txt")
t = data[:, 0]
y = data[:, 1]

xk_final = ovo(t, y)
y_pred = model(t, *xk_final)

plt.scatter(t, y, color="blue", alpha=0.6, label="Datos observados")
plt.plot(t, y_pred, color="red", linewidth=2, label="Modelo ajustado OVO")
plt.savefig("figuras/ovo_osborne_cauchy.pdf", bbox_inches="tight")
plt.show()
# Bibliotecas esenciales
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tabulate import tabulate

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

def hess_f_i(t_i):
    H = np.zeros((11,11))
    for i in range(11):
        for j in range(11):
            H[i,j] = t_i**(i+j)
    return H

def mount_Idelta(fovo, faux_sorted, indices, delta, Idelta, types, m):
    k = 0
    for i in range(m):
        diff = abs(fovo - faux_sorted[i])
        if diff <= delta:
            Idelta[k] = indices[i]
            types[k] = 'ineq'
            k += 1
    return k

def compute_Bkj(H):
    Hs = 0.5*(H + H.T)
    eigs = np.linalg.eigvalsh(Hs)
    lambda_min = np.min(eigs)
    ajuste = max(0, -lambda_min + 1e-6)
    B = Hs + ajuste*np.eye(Hs.shape[0])
    return 0.5*(B + B.T)

def constraint_fun(var, g, B):
    d = var[:11]
    z = var[11]
    return z - (g.dot(d) + 0.5*d.dot(B.dot(d)))

def constraint_jac(var, g, B):
    d = var[:11]
    gradc = np.zeros(12)
    gradc[:11] = -(g + B.dot(d))
    gradc[11] = 1.0
    return gradc

def ovoqn(t, y):
    epsilon = 1e-8
    delta = 1e-3
    deltax = 0.1
    theta = 0.7
    q = 77
    max_iter = 200
    max_iterarmijo = 100

    m = len(t)
    xk = np.array([1.3, 0.65, 0.65, 0.7, 0.6, 3, 5, 7, 2, 4.5, 5.5])
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)
    types  = np.empty(m, dtype=object)
    
    header = ["f(xk)", "Iter", "IterArmijo", "Mk(d)", "ncons", "Idelta"]
    table = []

    iteracion = 0
    while iteracion < max_iter:
        iteracion += 1

        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)

        indices = np.argsort(faux)
        faux_sorted = np.sort(faux)
        fxk = faux_sorted[q]

        nconst = mount_Idelta(fxk, faux_sorted, indices, delta, Idelta, types, m)
        
        if nconst == 0:
            break

        grads = []
        Bkjs = []
        constr_types = []
        for r in range(nconst):
            ind = Idelta[r]
            g = np.zeros(11)
            grad_f_i(t[ind], y[ind], xk, g)
            H = hess_f_i(t[ind])
            Bkjs.append(compute_Bkj(H))
            grads.append(g)
            constr_types.append(types[r])

        x0 = np.zeros(12)
        bounds = [
            (max(-10 - xk[0], -deltax), min(10 - xk[0], deltax)),
            (max(-10 - xk[1], -deltax), min(10 - xk[1], deltax)),
            (max(-10 - xk[2], -deltax), min(10 - xk[2], deltax)),
            (max(-10 - xk[3], -deltax), min(10 - xk[3], deltax)),
            (max(-10 - xk[3], -deltax), min(10 - xk[3], deltax)),
            (max(-10 - xk[3], -deltax), min(10 - xk[3], deltax)),
            (max(-10 - xk[3], -deltax), min(10 - xk[3], deltax)),
            (max(-10 - xk[3], -deltax), min(10 - xk[3], deltax)),
            (max(-10 - xk[3], -deltax), min(10 - xk[3], deltax)),
            (max(-10 - xk[3], -deltax), min(10 - xk[3], deltax)),
            (max(-10 - xk[3], -deltax), min(10 - xk[3], deltax)),
            (None, 0.0)
        ]

        constraints = []
        for g, B, ctype in zip(grads, Bkjs, constr_types):
            constraints.append({
                'type': ctype,
                'fun': lambda var, g=g, B=B: constraint_fun(var, g, B),
                'jac': lambda var, g=g, B=B: constraint_jac(var, g, B)
            })

        res = minimize(lambda var: var[4], x0, method="SLSQP",
                       bounds=bounds, constraints=constraints)

        d_sol = res.x[:11]
        mkd = res.fun

        if abs(mkd) < epsilon:
            xk += d_sol
            break

        alpha = 1
        iter_armijo = 0
        x_trial = xk
        while iter_armijo < max_iterarmijo:
            iter_armijo += 1
            x_trial = xk + alpha * d_sol
            faux_trial = np.array([f_i(ti, yi, x_trial) for ti, yi in zip(t, y)])
            fxk_trial = np.sort(faux_trial)[q]
            if fxk_trial <= fxk + theta * alpha * mkd:
                break
            alpha *= 0.5

        xk = x_trial
        table.append([fxk, iteracion, iter_armijo, mkd, nconst, Idelta[:min(5, nconst)].tolist()])
        np.savetxt('txt/sol_osborne2_cn.txt',xk, fmt="%.6f")
        
    print(tabulate(table, headers=header, tablefmt="grid"))
    print("Solución final:", xk)
    return xk

data = np.loadtxt("txt/data_osborne2.txt")
t = data[:,0]
y = data[:,1]

xk_final = ovoqn(t, y)
y_pred = model(t, *xk_final)

plt.scatter(t, y, color="blue", label="Datos")
plt.plot(t, y_pred, color="red", label="Ajuste")
plt.savefig("figuras/osborne2_cn.pdf", bbox_inches="tight")
plt.show()
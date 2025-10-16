import numpy as np
from scipy.optimize import minimize
from tabulate import tabulate
import matplotlib.pyplot as plt
import time

def model(t, x0, x1, x2):
    return x0 * np.exp((-x1 * ((t - x2)**2)) / 2)

def f_i(t_i, y_i, x):
    return 0.5 * ((model(t_i, *x) - y_i) ** 2)

def grad_f_i(t_i, y_i, x, grad):
    x0, x1, x2 = x
    s = t_i - x2
    z = np.exp((-x1 * s**2) / 2)
    diff = model(t_i, *x) - y_i
    grad[0] = diff * z
    grad[1] = diff * (-0.5 * x0 * s**2 * z)
    grad[2] = diff * (x0 * x1 * s * z)
    return grad

def hess_f_i(t_i, y_i, x):
    grad = np.zeros(3)
    grad_f_i(t_i, y_i, x, grad)
    H = np.outer(grad, grad)
    return H

def mount_Idelta(fovo, faux_sorted, indices, delta, Idelta, types, m):
    k = 0
    for i in range(m):
        if abs(fovo - faux_sorted[i]) <= delta:
            Idelta[k] = indices[i]
            types[k] = 'ineq'
            k += 1
    return k

def compute_Bkj(H):
    Hs = 0.5 * (H + H.T)
    eigs = np.linalg.eigvalsh(Hs)
    lambda_min = np.min(eigs)
    ajuste = max(0, -lambda_min + 1e-6)
    B = Hs + ajuste * np.eye(Hs.shape[0])
    return 0.5 * (B + B.T)

def constraint_fun(var, g, B):
    d = var[:3]
    z = var[3]
    return z - (g.dot(d) + 0.5 * d.dot(B.dot(d)))

def constraint_jac(var, g, B):
    d = var[:3]
    gradc = np.zeros(4)
    gradc[:3] = -(g + B.dot(d))
    gradc[3] = 1.0
    return gradc

def ovoqn(t, y, q):
    epsilon = 1e-11
    delta = 1e-4
    deltax = 1
    theta = 0.2
    max_iter = 200
    max_iterarmijo = 50

    m = len(t)
    xk = np.array([0.5, 1.0, 0.0])
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)
    types = np.empty(m, dtype=object)
    
    fcnt = 0
    iteracion = 0
    
    while iteracion < max_iter:
        iteracion += 1

        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)
        fcnt += m

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
            g = np.zeros(3)
            grad_f_i(t[ind], y[ind], xk, g)
            H = hess_f_i(t[ind], y[ind], xk)
            Bkjs.append(compute_Bkj(H))
            grads.append(g)
            constr_types.append(types[r])

        x0 = np.zeros(4)
        bounds = [(-deltax, deltax),
                  (-deltax, deltax),
                  (-deltax, deltax),
                  (None, 0.0)]

        constraints = []
        for g, B, ctype in zip(grads, Bkjs, constr_types):
            g_val = g.copy()
            B_val = B.copy()
            constraints.append({
                'type': ctype,
                'fun': lambda var, g=g_val, B=B_val: constraint_fun(var, g, B),
                'jac': lambda var, g=g_val, B=B_val: constraint_jac(var, g, B)
            })

        res = minimize(lambda var: var[3], x0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={'ftol': 1e-9, 'maxiter': 30, 'disp': False})

        d_sol = res.x[:3]
        mkd = float(res.fun)

        if abs(mkd) < epsilon:
            xk += d_sol
            break
        
        alpha = 1
        iter_armijo = 0
        while iter_armijo < max_iterarmijo:
            iter_armijo += 1
            x_trial = xk + alpha * d_sol
            faux_trial = np.array([f_i(ti, yi, x_trial) for ti, yi in zip(t, y)])
            fcnt += m
            fxk_trial = np.sort(faux_trial)[q]
            if fxk_trial <= fxk + theta * alpha * mkd:
                break
            alpha *= 0.5
            
        xk = x_trial

    return xk, fxk, iteracion, fcnt

data = np.loadtxt("data_gauss.txt")
t = data[:, 0]
y = data[:, 1]

m = len(t)
results = []
outliers_list = []
f_values = []

for o in range(4):
    q = m - o - 1
    
    start_time = time.time()
    xk_final, fxk, num_iter, fcnt = ovoqn(t, y, q)
    elapsed_time = time.time() - start_time
    
    results.append([o, fxk, num_iter, fcnt, elapsed_time, xk_final[0], xk_final[1], xk_final[2]])
    outliers_list.append(o)
    f_values.append(fxk)

headers = ["o", "f(x*)", "#it", "#fcnt", "Time (s)", "x0", "x1", "x2"]
print(tabulate(results, headers=headers, tablefmt="grid", floatfmt=(".0f", ".6e", ".0f", ".0f", ".6f", ".6f", ".6f", ".6f")))

plt.plot(outliers_list, f_values, 'o-', linewidth=2, markersize=8, color='blue')
plt.xlabel('Número de outliers ignorados', fontsize=12)
plt.ylabel('f(x*)', fontsize=12)
plt.savefig("figuras/cngaussvsouts_ovoqn.pdf", bbox_inches='tight')
plt.show()
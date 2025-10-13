import numpy as np
from scipy.optimize import minimize
from tabulate import tabulate
import matplotlib.pyplot as plt
import time

def model(t, x1, x2, x3, x4):
    return x1 + x2*t + x3*(t**2) + x4*(t**3)

def f_i(ti, yi, x):
    return 0.5 * (model(ti, *x) - yi)**2

def grad_f_i(ti, yi, x, grad):
    diff = model(ti, *x) - yi
    grad[0] = diff
    grad[1] = diff * ti
    grad[2] = diff * (ti**2)
    grad[3] = diff * (ti**3)
    return grad[:]

def hess_f_i(ti):
    H = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            H[i,j] = ti**(i+j)
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
    d = var[:4]
    z = var[4]
    return z - (g.dot(d) + 0.5*d.dot(B.dot(d)))

def constraint_jac(var, g, B):
    d = var[:4]
    gradc = np.zeros(5)
    gradc[:4] = -(g + B.dot(d))
    gradc[4] = 1.0
    return gradc

def ovoqn(t, y, q_value):
    epsilon = 1e-9
    delta = 1e-2
    deltax = 1.2
    theta = 0.7
    q = q_value
    max_iter = 200
    max_iterarmijo = 100

    m = len(t)
    xk = np.array([-1.0, -2.0, 1.0, -1.0])
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)
    types  = np.empty(m, dtype=object)
    
    fcnt = 0
    start_time = time.time()
    
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
            g = np.zeros(4)
            grad_f_i(t[ind], y[ind], xk, g)
            H = hess_f_i(t[ind])
            Bkjs.append(compute_Bkj(H))
            grads.append(g)
            constr_types.append(types[r])

        x0 = np.zeros(5)
        bounds = [
            (max(-10 - xk[0], -deltax), min(10 - xk[0], deltax)),
            (max(-10 - xk[1], -deltax), min(10 - xk[1], deltax)),
            (max(-10 - xk[2], -deltax), min(10 - xk[2], deltax)),
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

        d_sol = res.x[:4]
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
            fcnt += m
            fxk_trial = np.sort(faux_trial)[q]
            if fxk_trial <= fxk + theta * alpha * mkd:
                break
            alpha *= 0.5

        xk = x_trial

    elapsed_time = time.time() - start_time
    return xk, fxk, iteracion, fcnt, elapsed_time

data = np.loadtxt("data.txt")
t = data[:,0]
y = data[:,1]
m = len(t)

num_outliers = list(range(0, 11))
results = []

for n_out in num_outliers:
    q_value = m - n_out - 1
    print(f"Ejecutando OVO Cuasi-Newton con {n_out} outliers (q={q_value})...")
    xk_final, fxk, n_iter, n_fcnt, exec_time = ovoqn(t, y, q_value)
    results.append([n_out, fxk, n_iter, n_fcnt, exec_time])
    print(f"  f(x*) = {fxk:.6f}, #it = {n_iter}, #fcnt = {n_fcnt}, Time = {exec_time:.4f}s")

# Mostrar tabla
headers = ["o", "f(x*)", "#it", "#fcnt", "Time (s)"]
print("\n" + "="*70)
print(tabulate(results, headers=headers, tablefmt="grid", floatfmt=(".0f", ".6f", ".0f", ".0f", ".4f")))
print("="*70)

# Extraer valores para graficar
f_values = [row[1] for row in results]

plt.plot(num_outliers, f_values, 'o-', linewidth=2, markersize=8)
plt.xlabel('Número de outliers activos')
plt.ylabel('f(x*)')
plt.xticks(num_outliers)
plt.show()
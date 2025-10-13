import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from tabulate import tabulate
import time

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

def ovo(t, y, q_value):
    stop = 2e+0
    epsilon = 1e+8
    delta = 1e+1
    theta = 0.3
    n = 4
    m = len(t)
    q = q_value
    max_iter = 100
    max_iter_armijo = 30
    iter = 1
    fcnt = 0

    xk = np.array([0.02, 4000.0, 250.0])
    xktrial = np.zeros(3)
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)

    c = np.zeros(n)
    c[-1] = 1
    
    start_time = time.time()

    while iter <= max_iter:
        iter_armijo = 0

        x0_bounds = (-delta, delta)
        x1_bounds = (-delta, delta)
        x2_bounds = (-delta, delta)
        x3_bounds = (None, 0)

        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)
        
        fcnt += m

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
            
            fcnt += m
            
            faux_trial_sorted = np.sort(faux_trial)
            fxktrial = faux_trial_sorted[q]
            
            if fxktrial <= fxk + (theta * alpha * mkd):
                break
            alpha *= 0.5

        xk = xktrial
        iter += 1

    elapsed_time = time.time() - start_time
    return xk, fxk, iter - 1, fcnt, elapsed_time

# Cargar datos
data = np.loadtxt('data_meyer.txt')
t = data[:, 0]
y = data[:, 1]

# Ejecutar OVO para diferentes números de outliers
num_outliers = [0, 1, 2, 3]
m = len(t)
results = []

for n_out in num_outliers:
    q_value = m - n_out - 1
    print(f"Ejecutando OVO con {n_out} outliers (q={q_value})...")
    xk_final, fxk, n_iter, n_fcnt, exec_time = ovo(t, y, q_value)
    results.append([n_out, fxk, n_iter, n_fcnt, exec_time])
    print(f"  f(x*) = {fxk:.6f}, #it = {n_iter}, #fcnt = {n_fcnt}, Time = {exec_time:.4f}s")

# Mostrar tabla
headers = ["o", "f(x*)", "#it", "#fcnt", "Time (s)"]
print("\n" + "="*70)
print(tabulate(results, headers=headers, tablefmt="grid", floatfmt=(".0f", ".6f", ".0f", ".0f", ".7f")))
print("="*70)

# Extraer valores para graficar
f_values = [row[1] for row in results]

# Graficar
plt.plot(num_outliers, f_values, 'o-', linewidth=2, markersize=8)
plt.xlabel('Número de outliers activos')
plt.ylabel('f(x*)')
plt.xticks(num_outliers)
plt.savefig("figuras/meyervsouts.pdf", bbox_inches = 'tight')
plt.show()
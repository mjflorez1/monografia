import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from tabulate import tabulate
import time

def model(t,x1,x2,x3,x4):
    res = x1 + (x2 * t) + x3 * (t**2) + x4 * (t**3)
    return res

def f_i(t_i,y_i,x):
    res = 0.5 * ((model(t_i,*x) - y_i)**2)
    return res

def grad_f_i(t_i,y_i,x,grad):
    diff = model(t_i,*x) - y_i
    
    grad[0] = diff * 1
    grad[1] = diff * t_i
    grad[2] = diff * t_i**2
    grad[3] = diff * t_i**3

    return grad[:]

def mount_Idelta(fovo,faux,indices,delta,Idelta,m):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

def ovo_algorithm(t,y,q_value):
    
    m = len(t)
    epsilon = 1e-8
    delta   = 1e-3
    deltax  = 1.0
    theta   = 0.5
    n = 5
    q = q_value
    max_iter = 1000
    max_iter_armijo = 100
    iter = 1
    fcnt = 0

    xk = np.array([-1,-2,1,-1])
    xktrial = np.zeros(n-1)
    faux    = np.zeros(m)
    Idelta  = np.zeros(m,dtype=int)

    c = np.zeros(n)
    c[-1] = 1

    start_time = time.time()

    while iter <= max_iter:    

        iter_armijo = 0

        x0_bounds = (max(-10 - xk[0], -deltax), min(10 - xk[0], deltax))
        x1_bounds = (max(-10 - xk[1], -deltax), min(10 - xk[1], deltax))
        x2_bounds = (max(-10 - xk[2], -deltax), min(10 - xk[2], deltax))
        x3_bounds = (max(-10 - xk[3], -deltax), min(10 - xk[3], deltax))
        x4_bounds = (None, 0)

        for i in range(m):
            faux[i] = f_i(t[i],y[i],xk)
        
        fcnt += m

        indices = np.argsort(faux)
        faux = np.sort(faux)
        fxk = faux[q]

        nconst = mount_Idelta(fxk,faux,indices,delta,Idelta,m)

        A = np.zeros((nconst,n))
        b = np.zeros(nconst)
        grad = np.zeros((nconst,n-1))

        for i in range(nconst):
            ind = Idelta[i]
            grad_f_i(t[ind],y[ind],xk,grad[i,:])

            A[i,:-1] = grad[i,:]
            A[i,-1] = -1

        res = linprog(c,A_ub=A,b_ub=b,bounds=[x0_bounds,x1_bounds,x2_bounds,x3_bounds,x4_bounds])

        dk = res.x
        mkd = dk[-1]

        if abs(mkd) < epsilon:
            break

        alpha = 1

        while iter_armijo <= max_iter_armijo:

            iter_armijo += 1
            
            xktrial = xk + (alpha * dk[:-1])

            for i in range(m):
                faux[i] = f_i(t[i],y[i],xktrial)

            fcnt += m
            
            faux = np.sort(faux)
            fxktrial = faux[q]

            if fxktrial < fxk + (theta * alpha * mkd):
                break

            alpha = 0.5 * alpha

        xk = xktrial
        iter += 1

    elapsed_time = time.time() - start_time
    return xk, fxk, iter - 1, fcnt, elapsed_time

data = np.loadtxt("data.txt")
t = data[:,0]
y = data[:,1]
m = len(t)

num_outliers = list(range(0, 11))
results = []

for n_out in num_outliers:
    q_value = m - n_out - 1
    print(f"Ejecutando OVO con {n_out} outliers (q={q_value})...")
    xk_final, fxk, n_iter, n_fcnt, exec_time = ovo_algorithm(t, y, q_value)
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
plt.savefig("figuras/cauchyovo.pdf", bbox_inches = 'tight')
plt.show()
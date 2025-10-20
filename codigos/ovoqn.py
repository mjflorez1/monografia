"""
Algoritmo de Optimización por Valor de Orden (OVO) – Método quasi-Newton (SLSQP)
--------------------------------------------------------------------------------
Esta implementación está basada en el articulo:
  Andreani, R., Martínez, J. M., Salvatierra, M., & Yano, F. (2006).
  Quasi-Newton methods for order-value optimization and value-at-risk calculations.
  Pacific Journal of Optimization, 2(1), 11-33.
    
Resumen:
  Este código implementa un método iterativo para resolver el problema de Optimización por Valor de Orden (OVO),
  una generalización del problema clásico de Minimax. El objetivo es minimizar el valor funcional que ocupa la 
  posición p-ésima dentro de un conjunto dado de funciones. A diferencia del método tipo Cauchy, aquí el 
  subproblema se formula como un problema cuadrático y se resuelve usando el método SLSQP (Sequential Least Squares 
  Quadratic Programming), permitiendo una aproximación de tipo quasi-Newton mediante matrices Hessianas por cada restricción.
    
Implementación en Python realizada por:
  Mateo Florez  
  Email: mjflorez@mail.uniatlantico.edu.co
  Estudiante del programa de Matemáticas  
  Universidad del Atlántico

Orientador:
  Dr. Gustavo Quintero  
  Email: gdquintero@uniatlantico.edu.co
  Tutor de la monografía de grado  
  Universidad del Atlántico
"""

import numpy as np
from scipy.optimize import minimize
from tabulate import tabulate
import matplotlib.pyplot as plt

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

def ovoqn(t, y):
    epsilon = 1e-9
    delta = 1e-2
    deltax = 1.2
    theta = 0.7
    q = 35
    max_iter = 200
    max_iterarmijo = 100

    m = len(t)
    xk = np.array([-1.0, -2.0, 1.0, -1.0])
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
            fxk_trial = np.sort(faux_trial)[q]
            if fxk_trial <= fxk + theta * alpha * mkd:
                break
            alpha *= 0.5

        xk = x_trial
        table.append([fxk, iteracion, iter_armijo, mkd, nconst, Idelta[:min(5, nconst)].tolist()])
        
    print(tabulate(table, headers=header, tablefmt="grid"))
    print("Solución final:", xk)
    return xk

# ---------------------- CARGA DE DATOS ----------------------
data = np.loadtxt("data.txt")
t = data[:,0]
y = data[:,1]
m = len(t)

# ---------------------- OVO ----------------------
xk_final = ovoqn(t,y)
y_ovo = model(t, *xk_final)

# ---------------------- MÉTODO minimize ----------------------
x_star = [0, 2, -3, 1]
w = x_star[0] + x_star[1]*t + x_star[2]*t**2 + x_star[3]*t**3

def objetivo(x):
    y_modelo = np.polyval(x[::-1], t)
    return np.sum((y - y_modelo) ** 2)

x0 = [-1, -2, 1, -1]
bounds = [(-10, 10)] * 4
res = minimize(objetivo, x0, method='L-BFGS-B', bounds=bounds)

print("\nCoeficientes mínimos cuadrados:")
print(res.x)
print(f"Iteraciones minimize: {res.nit}")
print(f"Valor final función objetivo: {res.fun:.6e}")

y_fit = np.polyval(res.x[::-1], t)

# ---------------------- GRÁFICA FINAL ----------------------
plt.figure()
plt.scatter(t, y, color='y', label='Datos observados')
plt.plot(t, y_fit, 'b-', linewidth=1.5, label='Ajuste con OLS')
plt.plot(t, y_ovo, 'r-', linewidth=1.5, label='Ajuste OVO')
plt.legend(loc='lower right')
plt.savefig("figuras/comparacion_ovoqn_ols.pdf", bbox_inches='tight')
plt.show()
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

# Bibliotecas importantes
import numpy as np
from scipy.optimize import minimize

# Modelo cúbico
def model(t, x1, x2, x3, x4):
    return x1 + (x2 * t) + x3 * (t**2) + x4 * (t**3)

# Función de error cuadrático
def f_i(t_i, y_i, x):
    return 0.5 * ((model(t_i, *x) - y_i)**2)

# Gradiente de la función de error
def grad_f_i(t_i, y_i, x, grad):
    diff = model(t_i, *x) - y_i
    grad[0] = diff
    grad[1] = diff * t_i
    grad[2] = diff * (t_i**2)
    grad[3] = diff * (t_i**3)
    return grad

# Hessiana
def hess_f_i(ti):
    phi = np.array([1.0, ti, ti**2, ti**3], dtype=float)
    return np.outer(phi, phi)

# Construir conjunto I_delta
def mount_Idelta(fovo, faux, indices, delta, Idelta, m):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

# Computar matriz B_kj
def compute_Bkj(H, epsilon=1e-8, reg=1e-12):
    Hs = 0.5 * (H + H.T)
    eigs = np.linalg.eigvalsh(Hs)
    lambda_min = np.min(eigs)
    ajuste = max(1e-3, -lambda_min + epsilon)
    B = Hs + ajuste * np.eye(Hs.shape[0])
    B += reg * np.eye(Hs.shape[0])
    return 0.5 * (B + B.T)

# Funciones de restricción para SLSQP
def constraint_fun(var, g, B):
    d = var[:4]
    z = var[4]
    return float(z - (g.dot(d) + 0.5 * d.dot(B.dot(d))))

def constraint_jac(var, g, B):
    d = var[:4]
    gradc = np.zeros(5, dtype=float)  # 4 parámetros + z
    gradc[:4] = -g - B.dot(d)
    gradc[4] = 1.0
    return gradc

# Algoritmo quasi-Newton (SLSQP)
def ovo_qnewton_slsqp(t, y):
    epsilon = 1e-8
    delta = 1e-3
    deltax = 1.0
    theta = 0.5
    q = 35
    max_iter = 500
    max_iter_armijo = 50

    m = len(t)
    q = min(q, m - 1)

    # Solución inicial
    xk = np.array([-1.0, -2.0, 1.0, -1.0], dtype=float)
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)

    iteracion = 1
    while iteracion <= max_iter:
        # Calcular valores de f_i
        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)

        indices = np.argsort(faux)
        faux_sorted = np.sort(faux)
        fxk = faux_sorted[q]

        nconst = mount_Idelta(fxk, faux_sorted, indices, delta, Idelta, m)
        if nconst == 0:
            break

        # Gradientes y matrices B_kj
        grads = []
        Bkjs = []
        for r in range(nconst):
            ind = Idelta[r]
            g = np.zeros(4)
            grad_f_i(t[ind], y[ind], xk, g)
            H = hess_f_i(t[ind])
            Bkj = compute_Bkj(H)
            grads.append(g.copy())
            Bkjs.append(Bkj.copy())

        # Definición del problema SLSQP
        nv = 5  # 4 variables del modelo + z
        def obj(var):
            return float(var[-1])
        def obj_jac(var):
            grad_obj = np.zeros(nv, dtype=float)
            grad_obj[-1] = 1.0
            return grad_obj

        # Restricciones de desigualdad
        cons = []
        for g_local, B_local in zip(grads, Bkjs):
            cons.append({
                'type': 'ineq',
                'fun': lambda var, g=g_local, B=B_local: constraint_fun(var, g, B),
                'jac': lambda var, g=g_local, B=B_local: constraint_jac(var, g, B)
            })

        # Restricciones de caja
        bounds = [(max(-10.0 - xk[i], -deltax), min(10.0 - xk[i], deltax)) for i in range(4)]
        bounds.append((None, 0.0))  # z ≤ 0

        # Punto inicial
        x0 = np.zeros(nv, dtype=float)
        x0[4] = -0.1

        res = minimize(obj, x0, method='SLSQP', jac=obj_jac,
                       bounds=bounds, constraints=cons,
                       options={'ftol': 1e-9, 'maxiter': 200})

        # Solución del subproblema convexo
        d_sol = res.x[:4]
        mkd = float(res.fun)

        # Criterio de parada
        if abs(mkd) < epsilon or np.linalg.norm(d_sol) < epsilon:
            xk += d_sol
            break
        if mkd >= -1e-12:
            break

        # Búsqueda de línea (Armijo)
        iter_armijo = 0
        alpha = 1.0
        while iter_armijo < max_iter_armijo:
            iter_armijo += 1
            x_trial = xk + alpha * d_sol
            faux_trial = np.array([f_i(ti, yi, x_trial) for ti, yi in zip(t, y)])
            fxk_trial = np.sort(faux_trial)[q]
            if fxk_trial <= fxk + theta * alpha * mkd:
                break
            alpha *= 0.5
            
        print(iteracion,fxk,mkd,iter_armijo)

        xk = x_trial.copy()
        iteracion += 1

    print("Solución final:", xk)
    return xk

data = np.loadtxt("data.txt")
t = data[:, 0]
y = data[:, 1]

ovo_qnewton_slsqp(t, y)
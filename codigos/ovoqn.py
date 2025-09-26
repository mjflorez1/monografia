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
import pandas as pd
import time

# Modelo cúbico
def model(t, x1, x2, x3, x4):
    return x1 + x2*t + x3*(t**2) + x4*(t**3)

# Función de error y gradiente
def f_i(ti, yi, x):
    return 0.5 * (model(ti, *x) - yi)**2

def grad_f_i(ti, yi, x, grad):
    diff = model(ti, *x) - yi
    grad[0] = diff
    grad[1] = diff * ti
    grad[2] = diff * (ti**2)
    grad[3] = diff * (ti**3)
    return grad

# Hessiana
def hess_f_i(ti):
    H = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            H[i,j] = ti**(i+j)
    return H

# Conjunto I_delta
def mount_Idelta(fovo, faux, indices, delta, Idelta, types, m):
    k = 0
    for i in range(m):
        diff = abs(fovo - faux[i])
        if diff <= delta:
            Idelta[k] = indices[i]
            # igualdad si está muy cerca, desigualdad en otro caso
            if diff < delta/2:
                types[k] = 'eq'
            else:
                types[k] = 'ineq'
            k += 1
    return k

# Construcción de B_kj
def compute_Bkj(H, first_iter=False):
    if first_iter:
        return np.eye(H.shape[0])
    Hs = 0.5*(H + H.T)
    eigs = np.linalg.eigvalsh(Hs)
    lambda_min = np.min(eigs)
    ajuste = max(0, -lambda_min + 1e-12)
    B = Hs + ajuste*np.eye(Hs.shape[0])
    return 0.5*(B + B.T)

# Constraints
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

# OVO tipo quasi-Newton modificado para devolver métricas
def ovoqn_metrics(t, y, o_value):
    epsilon = 1e-15
    delta = 1e-4
    deltax = 1.0
    theta = 0.9
    max_iter = 2000
    max_iterarmijo = 150

    m = len(t)
    o = min(o_value, m-1)
    xk = np.array([-1.0, -2.0, 1.0, -1.0])
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)
    types  = np.empty(m, dtype=object)

    iteracion = 0
    total_fcnt = 0
    start_time = time.time()
    
    while iteracion < max_iter:
        iteracion += 1

        # Evaluación de función
        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)
        total_fcnt += m

        indices = np.argsort(faux)
        faux_sorted = np.sort(faux)
        fxk = faux_sorted[o]

        # Construcción de I_delta
        nconst = mount_Idelta(fxk, faux_sorted, indices, delta, Idelta, types, m)
        if nconst == 0:
            break

        # Limitar el número máximo de restricciones para evitar problemas con SLSQP
        max_constraints = 50
        if nconst > max_constraints:
            nconst = max_constraints

        # Se calcula el gradiente y la hessiana
        grads = []
        Bkjs = []
        constr_types = []
        for r in range(nconst):
            ind = Idelta[r]
            g = np.zeros(4)
            grad_f_i(t[ind], y[ind], xk, g)
            H = hess_f_i(t[ind])
            Bkjs.append(compute_Bkj(H, first_iter=(iteracion==1)))
            grads.append(g)
            constr_types.append(types[r])

        # Subproblema cuadrático
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

        try:
            # Minimizador con tolerancias estrictas
            res = minimize(lambda var: var[4], x0, method="SLSQP",
                           bounds=bounds, constraints=constraints,
                           options={'ftol':1e-12, 'eps':1e-10, 'maxiter':200, 'disp':False})

            if not res.success:
                break

            d_sol = res.x[:4]
            mkd = float(res.fun)

            # Criterio de parada
            if abs(mkd) < epsilon or np.linalg.norm(d_sol) < 1e-10:
                xk += d_sol
                break
            if mkd >= -1e-14:
                break

            # Armijo
            alpha = 1.0
            iter_armijo = 0
            while iter_armijo < max_iterarmijo:
                iter_armijo += 1
                x_trial = xk + alpha * d_sol
                faux_trial = np.array([f_i(ti, yi, x_trial) for ti, yi in zip(t, y)])
                total_fcnt += m
                fxk_trial = np.sort(faux_trial)[o]
                if fxk_trial <= fxk + theta * alpha * mkd:
                    break
                alpha *= 0.5

            xk = x_trial

            # Parada adicional por progreso mínimo
            if iteracion > 50 and abs(mkd) < 1e-10:
                break

        except Exception as e:
            break

    end_time = time.time()
    execution_time = end_time - start_time
    
    # Calcular f(x*) final
    faux_final = np.array([f_i(ti, yi, xk) for ti, yi in zip(t, y)])
    total_fcnt += m
    f_final = np.sort(faux_final)[o]
    
    return {
        'o': o_value,
        'f(x*)': f_final,
        '#it': iteracion,
        '#fcnt': total_fcnt,
        'Time': execution_time
    }

# Carga de datos
data = np.loadtxt("data.txt")
t = data[:,0]
y = data[:,1]
print("Datos cargados desde data.txt")

# Ejecutar solo para o = 0 a 10
o_values = list(range(0, 11))  # o de 0 a 10

print("Ejecutando algoritmo OVO-QN...")
print("=" * 70)

results = []
for o in o_values:
    print(f"Ejecutando para o = {o}...")
    result = ovoqn_metrics(t, y, o)
    results.append(result)
    
    # Formatear el valor f(x*) para mostrar
    f_val = result['f(x*)']
    if f_val < 1e-15:
        f_str = f"{f_val:.3e}"
    else:
        f_str = f"{f_val:.6e}"
    
    print(f"Resultado: o={o}, f(x*)={f_str}, #it={result['#it']}, #fcnt={result['#fcnt']}, Time={result['Time']:.4f}s")
    print("-" * 70)

# Crear DataFrame con las columnas específicas
df = pd.DataFrame(results, columns=["o", "f(x*)", "#it", "#fcnt", "Time"])

# Formatear los valores para mejor presentación
def format_scientific(x):
    if abs(x) < 1e-100:
        return "0"
    elif abs(x) < 1e-15:
        return f"{x:.3e}"
    else:
        exponent = int(np.floor(np.log10(abs(x))))
        coefficient = x / (10 ** exponent)
        return f"{coefficient:.6f} × 10^{{{exponent}}}"

df['f(x*)'] = df['f(x*)'].apply(format_scientific)
df['Time'] = df['Time'].apply(lambda x: f"{x:.6f}")

print("\n" + "=" * 70)
print("RESULTADOS FINALES - ALGORITMO OVO-QN")
print("=" * 70)
print(df.to_string(index=False))
print("=" * 70)
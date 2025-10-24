import numpy as np
from scipy.optimize import linprog, minimize
from tabulate import tabulate
import time
import os

# ============= MODELO Y FUNCIONES COMUNES =============
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

# ============= ALGORITMO CAUCHY =============
def mount_Idelta_cauchy(fovo, faux, indices, delta, Idelta, m):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

def ovo_cauchy(t, y, q):
    epsilon = 1e-8
    delta = 1e-3
    deltax = 1.0
    theta = 0.5
    n = 5
    max_iter = 1000
    max_iter_armijo = 100
    
    m = len(t)
    xk = np.array([-1.0, -2.0, 1.0, -1.0])
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)
    c = np.zeros(n)
    c[-1] = 1
    
    iter_count = 0
    fcnt_total = 0
    
    start_time = time.time()
    
    while iter_count < max_iter:
        iter_count += 1
        
        x0_bounds = (max(-10 - xk[0], -deltax), min(10 - xk[0], deltax))
        x1_bounds = (max(-10 - xk[1], -deltax), min(10 - xk[1], deltax))
        x2_bounds = (max(-10 - xk[2], -deltax), min(10 - xk[2], deltax))
        x3_bounds = (max(-10 - xk[3], -deltax), min(10 - xk[3], deltax))
        x4_bounds = (None, 0)
        
        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)
        
        indices = np.argsort(faux)
        faux_sorted = np.sort(faux)
        fxk = faux_sorted[q]
        
        nconst = mount_Idelta_cauchy(fxk, faux_sorted, indices, delta, Idelta, m)
        
        if nconst == 0:
            break
        
        A = np.zeros((nconst, n))
        b = np.zeros(nconst)
        grad = np.zeros((nconst, n-1))
        
        for i in range(nconst):
            ind = Idelta[i]
            grad_f_i(t[ind], y[ind], xk, grad[i, :])
            A[i, :-1] = grad[i, :]
            A[i, -1] = -1
        
        res = linprog(c, A_ub=A, b_ub=b, 
                     bounds=[x0_bounds, x1_bounds, x2_bounds, x3_bounds, x4_bounds],
                     method='highs')
        
        dk = res.x
        mkd = dk[-1]
        
        if abs(mkd) < epsilon:
            break
        
        alpha = 1
        iter_armijo = 0
        
        while iter_armijo < max_iter_armijo:
            iter_armijo += 1
            fcnt_total += 1
            
            xktrial = xk + alpha * dk[:-1]
            
            for i in range(m):
                faux[i] = f_i(t[i], y[i], xktrial)
            
            faux_sorted = np.sort(faux)
            fxktrial = faux_sorted[q]
            
            if fxktrial < fxk + theta * alpha * mkd:
                break
            
            alpha = 0.5 * alpha
        
        xk = xktrial
    
    elapsed_time = time.time() - start_time
    
    # Calcular f(x*) final
    for i in range(m):
        faux[i] = f_i(t[i], y[i], xk)
    faux_sorted = np.sort(faux)
    f_final = faux_sorted[q]
    
    return f_final, iter_count, fcnt_total, elapsed_time, xk

# ============= ALGORITMO CUASI-NEWTON =============
def hess_f_i(ti):
    H = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            H[i, j] = ti**(i+j)
    return H

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

def mount_Idelta_cn(fovo, faux_sorted, indices, delta, Idelta, types, m):
    k = 0
    for i in range(m):
        diff = abs(fovo - faux_sorted[i])
        if diff <= delta:
            Idelta[k] = indices[i]
            types[k] = 'ineq'
            k += 1
    return k

def ovo_cuasinewton(t, y, q):
    epsilon = 1e-9
    delta = 1e-2
    deltax = 1.2
    theta = 0.7
    max_iter = 200
    max_iterarmijo = 100
    
    m = len(t)
    xk = np.array([-1.0, -2.0, 1.0, -1.0])
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)
    types = np.empty(m, dtype=object)
    
    iter_count = 0
    fcnt_total = 0
    
    start_time = time.time()
    
    while iter_count < max_iter:
        iter_count += 1
        
        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)
        
        indices = np.argsort(faux)
        faux_sorted = np.sort(faux)
        fxk = faux_sorted[q]
        
        nconst = mount_Idelta_cn(fxk, faux_sorted, indices, delta, Idelta, types, m)
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
            fcnt_total += 1
            
            x_trial = xk + alpha * d_sol
            faux_trial = np.array([f_i(ti, yi, x_trial) for ti, yi in zip(t, y)])
            fxk_trial = np.sort(faux_trial)[q]
            
            if fxk_trial <= fxk + theta * alpha * mkd:
                break
            alpha *= 0.5
        
        xk = x_trial
    
    elapsed_time = time.time() - start_time
    
    # Calcular f(x*) final
    for i in range(m):
        faux[i] = f_i(t[i], y[i], xk)
    faux_sorted = np.sort(faux)
    f_final = faux_sorted[q]
    
    return f_final, iter_count, fcnt_total, elapsed_time, xk

# ============= FUNCIÓN PRINCIPAL =============
def run_comparison():
    # Cargar datos
    data = np.loadtxt("txt/data.txt")
    t = data[:, 0]
    y = data[:, 1]
    m = len(t)
    
    # Tabla de resultados
    results = []
    
    # Probar con diferentes números de outliers
    outliers_list = range(0, 16)  # De 0 a 15
    
    print("Ejecutando comparación...")
    print("=" * 80)
    
    for num_outliers in outliers_list:
        q = m - num_outliers - 1  # Índice del q-ésimo valor ordenado
        
        if q < 0 or q >= m:
            continue
        
        print(f"\nProbando con {num_outliers} outliers (q={q})...")
        
        # Ejecutar Cauchy
        try:
            f_cauchy, it_cauchy, fcnt_cauchy, time_cauchy, _ = ovo_cauchy(t, y, q)
        except Exception as e:
            print(f"Error en Cauchy: {e}")
            f_cauchy, it_cauchy, fcnt_cauchy, time_cauchy = np.nan, np.nan, np.nan, np.nan
        
        # Ejecutar Cuasi-Newton
        try:
            f_cn, it_cn, fcnt_cn, time_cn, _ = ovo_cuasinewton(t, y, q)
        except Exception as e:
            print(f"Error en Cuasi-Newton: {e}")
            f_cn, it_cn, fcnt_cn, time_cn = np.nan, np.nan, np.nan, np.nan
        
        results.append([
            num_outliers,
            f"{f_cauchy:.6e}" if not np.isnan(f_cauchy) else "N/A",
            it_cauchy if not np.isnan(it_cauchy) else "N/A",
            fcnt_cauchy if not np.isnan(fcnt_cauchy) else "N/A",
            f"{time_cauchy:.4f}" if not np.isnan(time_cauchy) else "N/A",
            f"{f_cn:.6e}" if not np.isnan(f_cn) else "N/A",
            it_cn if not np.isnan(it_cn) else "N/A",
            fcnt_cn if not np.isnan(fcnt_cn) else "N/A",
            f"{time_cn:.4f}" if not np.isnan(time_cn) else "N/A"
        ])
    
    # Imprimir tabla
    headers = ["o", "f(x*) Cauchy", "#it", "#fcnt", "time", 
               "f(x*) CN", "#it", "#fcnt", "time"]
    
    print("\n" + "=" * 80)
    print("RESULTADOS FINALES")
    print("=" * 80)
    print(tabulate(results, headers=headers, tablefmt="grid"))
    
    # Guardar resultados
    os.makedirs('txt', exist_ok=True)
    with open('txt/comparison_results.txt', 'w') as f:
        f.write(tabulate(results, headers=headers, tablefmt="grid"))
    
    print("\nResultados guardados en 'txt/comparison_results.txt'")

if __name__ == "__main__":
    run_comparison()
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tabulate import tabulate

def model(t_i, x0, x1, x2):
    return x0 * np.exp(x1 / (t_i + x2))

def f_i(ti, yi, x):
    return 0.5 * ((model(ti, *x) - yi)**2)

def grad_f_i(ti, yi, x, grad):
    x0, x1, x2 = x
    z = np.exp(x1 / (ti + x2))
    diff = (x0 * z) - yi
    grad[0] = diff * z
    grad[1] = diff * x0 * z / (ti + x2)
    grad[2] = diff * (-x0 * x1 * z / ((ti + x2)**2))
    return grad[:]

def hess_f_i(ti, yi, x):
    x0, x1, x2 = x
    t2 = ti + x2
    exp_val = np.exp(x1 / t2)
    r = model(ti, x0, x1, x2) - yi
    
    # Gradientes
    g0 = exp_val
    g1 = x0 * exp_val / t2
    g2 = -x0 * x1 * exp_val / (t2**2)
    
    # Hessiano exacto
    H = np.zeros((3, 3))
    H[0,0] = g0 * g0  # d²f/dx0²
    H[0,1] = g0 * g1 + r * (exp_val / t2)  # d²f/dx0dx1
    H[0,2] = g0 * g2 + r * (-x1 * exp_val / (t2**2))  # d²f/dx0dx2
    
    H[1,0] = H[0,1]
    H[1,1] = g1 * g1 + r * (x0 * exp_val / (t2**2))  # d²f/dx1²
    H[1,2] = g1 * g2 + r * (-x0 * exp_val / (t2**2) - x0 * x1 * exp_val / (t2**3))  # d²f/dx1dx2
    
    H[2,0] = H[0,2]
    H[2,1] = H[1,2]
    H[2,2] = g2 * g2 + r * (2 * x0 * x1 * exp_val / (t2**3) + x0 * x1**2 * exp_val / (t2**4))  # d²f/dx2²
    
    return H

def mount_Idelta(fovo, faux, indices, delta, Idelta, types, m):
    k = 0
    for i in range(m):
        diff = abs(fovo - faux[i])
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
    d = var[:3]
    z = var[3]
    return z - (g.dot(d) + 0.5*d.dot(B.dot(d)))

def constraint_jac(var, g, B):
    d = var[:3]
    gradc = np.zeros(4)
    gradc[:3] = -(g + B.dot(d))
    gradc[3] = 1.0
    return gradc

def ovoqn(t, y):
    epsilon = 1e-3
    delta = 1e+4  # REDUCIDO de 1e+7 a 1e+3
    deltax = 100
    theta = 0.01
    q = 12
    max_iter = 200
    max_iterarmijo = 50

    m = len(t)
    xk = np.array([0.02, 4000, 250])
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)
    types = np.empty(m, dtype=object)
    
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
            g = np.zeros(3)
            grad_f_i(t[ind], y[ind], xk, g)
            H = hess_f_i(t[ind], y[ind], xk)  # AHORA USA EL HESSIANO CORRECTO
            Bkjs.append(compute_Bkj(H))
            grads.append(g)
            constr_types.append(types[r])

        # Calcular gradiente de la función OVO para Armijo
        grad_ovo = np.zeros(3)
        count = 0
        for i in range(m):
            if abs(faux[i] - fxk) <= delta:
                g_temp = np.zeros(3)
                grad_f_i(t[i], y[i], xk, g_temp)
                grad_ovo += g_temp
                count += 1
        if count > 0:
            grad_ovo /= count

        x0 = np.zeros(4)
        bounds = [(-deltax, deltax),
                  (-deltax, deltax),
                  (-deltax, deltax),
                  (None, 0.0)]

        constraints = []
        # CORREGIDO: problema de late binding en lambdas
        for i, (g, B, ctype) in enumerate(zip(grads, Bkjs, constr_types)):
            constraints.append({
                'type': ctype,
                'fun': lambda var, g_i=g, B_i=B: constraint_fun(var, g_i, B_i),
                'jac': lambda var, g_i=g, B_i=B: constraint_jac(var, g_i, B_i)
            })

        res = minimize(lambda var: var[3], x0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={'ftol':1e-9, 'maxiter':100})

        d_sol = res.x[:3]
        mkd = float(res.fun)

        if abs(mkd) < epsilon:
            xk += d_sol
            break
        
        # CORREGIDO: condición de Armijo con gradiente real
        alpha = 1
        iter_armijo = 0
        grad_dot_d = np.dot(grad_ovo, d_sol)
        
        while iter_armijo < max_iterarmijo:
            iter_armijo += 1
            x_trial = xk + alpha * d_sol
            faux_trial = np.array([f_i(ti, yi, x_trial) for ti, yi in zip(t, y)])
            fxk_trial = np.sort(faux_trial)[q]
            
            # Usa grad_dot_d en lugar de mkd
            if fxk_trial <= fxk + theta * alpha * grad_dot_d:
                xk = x_trial
                break
            
            alpha *= 0.5
        
        table.append([fxk, iteracion, iter_armijo, mkd, nconst, Idelta[:min(5, nconst)].tolist()])

    print(tabulate(table, headers=header, tablefmt="grid"))
    print("Solución final:", xk)
    return xk

data = np.loadtxt("data_meyer.txt")
t = data[:,0]
y = data[:,1]

xk_final = ovoqn(t, y)
y_pred = model(t, *xk_final)

plt.scatter(t, y, color="magenta", alpha=0.6, label="Datos observados")
plt.plot(t, y_pred, color="green", linewidth=2, label="Modelo ajustado OVOQN")
plt.legend()
plt.xlabel("t")
plt.ylabel("y")
plt.title("Ajuste del modelo de Meyer usando OVOQN")
plt.show()
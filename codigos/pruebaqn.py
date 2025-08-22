import numpy as np
from scipy.optimize import minimize

# Modelo cúbico
def model(t, x1, x2, x3, x4):
    return x1 + (x2 * t) + x3 * (t**2) + x4 * (t**3)

# Función de error cuadrático
def f_i(t_i, y_i, x):
    return 0.5 * ((model(t_i, *x) - y_i)**2)

# Gradiente de la función de error
def grad_f_i(t_i, y_i, x):
    diff = model(t_i, *x) - y_i
    return np.array([diff, diff*t_i, diff*(t_i**2), diff*(t_i**3)], dtype=float)

# Hessiana
def hess_f_i(ti):
    phi = np.array([1.0, ti, ti**2, ti**3], dtype=float)
    return np.outer(phi, phi)

# Construir conjunto I_delta
def mount_Idelta(fovo, faux, indices, delta, m):
    Idelta = []
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta.append(indices[i])
    return Idelta

# Computar matriz B_kj usando descomposición espectral
def compute_Bkj_spectral(H, epsilon=1e-8, reg=1e-8):
    Hs = 0.5 * (H + H.T)
    eigvals, eigvecs = np.linalg.eigh(Hs)
    eigvals_adj = np.maximum(eigvals, epsilon)
    B = eigvecs @ np.diag(eigvals_adj) @ eigvecs.T
    B += reg * np.eye(H.shape[0])
    return 0.5 * (B + B.T)

# Funciones de restricción para SLSQP
def constraint_fun(var, g, B):
    d = var[:4]
    z = var[4]
    return float(z - (g.dot(d) + 0.5 * d.dot(B.dot(d))))

def constraint_jac(var, g, B):
    d = var[:4]
    gradc = np.zeros(5, dtype=float)
    gradc[:4] = -g - B.dot(d)
    gradc[4] = 1.0
    return gradc

# Función objetivo y gradiente (fuera del while)
nv = 5
def obj(var):
    return float(var[-1])

def obj_jac(var):
    grad_obj = np.zeros(nv, dtype=float)
    grad_obj[-1] = 1.0
    return grad_obj

# Ejemplo de restricción de igualdad: suma de x = 1
def eq_constraint(var):
    d = var[:4]
    return np.sum(d) - 1  # debe ser cero

def eq_constraint_jac(var):
    grad = np.zeros(5, dtype=float)
    grad[:4] = 1.0
    grad[4] = 0.0
    return grad

# Algoritmo quasi-Newton (SLSQP)
def ovoqn(t, y):
    epsilon = 1e-8
    delta = 1e-3
    deltax = 1.0
    theta = 0.01
    q = 35
    max_iter = 500
    max_iter_armijo = 20

    m = len(t)
    q = min(q, m - 1)
    xk = np.array([-1.0, -2.0, 1.0, -1.0], dtype=float)
    fxk_prev = np.inf

    iteracion = 1
    while iteracion <= max_iter:
        faux = np.array([f_i(ti, yi, xk) for ti, yi in zip(t, y)])
        indices = np.argsort(faux)
        faux_sorted = np.sort(faux)
        fxk = faux_sorted[q]

        Idelta = mount_Idelta(fxk, faux_sorted, indices, delta, m)
        nconst = len(Idelta)
        if nconst == 0:
            break

        grads = []
        Bkjs = []
        for ind in Idelta:
            g = grad_f_i(t[ind], y[ind], xk)
            H = hess_f_i(t[ind])
            Bkj = compute_Bkj_spectral(H)
            grads.append(g)
            Bkjs.append(Bkj)

        # Restricciones SLSQP
        cons = []
        # desigualdades de OVO
        for g_local, B_local in zip(grads, Bkjs):
            cons.append({
                'type': 'ineq',
                'fun': lambda var, g=g_local, B=B_local: constraint_fun(var, g, B),
                'jac': lambda var, g=g_local, B=B_local: constraint_jac(var, g, B)
            })
        # igualdad
        cons.append({
            'type': 'eq',
            'fun': eq_constraint,
            'jac': eq_constraint_jac
        })

        # Restricciones de caja
        bounds = [(max(-10.0 - xk[i], -deltax), min(10.0 - xk[i], deltax)) for i in range(4)]
        bounds.append((None, 0.0))

        x0 = np.zeros(nv, dtype=float)
        x0[4] = -0.1

        res = minimize(obj, x0, method='SLSQP', jac=obj_jac,
                       bounds=bounds, constraints=cons,
                       options={'ftol': 1e-9, 'maxiter': 200})

        d_sol = res.x[:4]
        mkd = float(res.fun)

        if abs(mkd) < epsilon or np.linalg.norm(d_sol) < epsilon:
            xk += d_sol
            break
        if abs(fxk - fxk_prev) < 1e-6:
            xk += d_sol
            break
        if mkd >= -1e-12:
            xk += d_sol
            break

        # Búsqueda de línea Armijo
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

        print(iteracion, fxk, mkd, iter_armijo)

        xk = x_trial
        fxk_prev = fxk
        iteracion += 1

    print("Solución final:", xk)
    return xk

# Cargar datos y ejecutar
data = np.loadtxt("data.txt")
t = data[:, 0]
y = data[:, 1]

ovoqn(t, y)
import numpy as np
from scipy.optimize import minimize

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
    ajuste = max(0, -lambda_min + 1e-8)
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

# OVO tipo quasi-Newton
def ovoqn(t, y):
    epsilon = 1e-8
    delta = 1e-2
    deltax = 1.0
    theta = 0.9
    q = 32
    max_iter = 200
    max_iterarmijo = 100

    m = len(t)
    q = min(q, m-1)
    xk = np.array([-1.0, -2.0, 1.0, -1.0])
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)
    types  = np.empty(m, dtype=object)

    iteracion = 0
    while iteracion < max_iter:
        iteracion += 1

        # Evaluación de función
        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)

        indices = np.argsort(faux)
        faux_sorted = np.sort(faux)
        fxk = faux_sorted[q]

        # Construcción de I_delta
        nconst = mount_Idelta(fxk, faux_sorted, indices, delta, Idelta, types, m)
        if nconst == 0:
            break

        # Se calcula el gradiente y la hessiana, se construye la matriz Bkj
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

        # restricciones eq/ineq
        constraints = []
        for g, B, ctype in zip(grads, Bkjs, constr_types):
            constraints.append({
                'type': ctype,
                'fun': lambda var, g=g, B=B: constraint_fun(var, g, B),
                'jac': lambda var, g=g, B=B: constraint_jac(var, g, B)
            })

        res = minimize(lambda var: var[4], x0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={'ftol':1e-8, 'maxiter':100, 'disp':False})

        d_sol = res.x[:4]
        mkd = float(res.fun)

        # Criterio de parada
        if abs(mkd)<epsilon or np.linalg.norm(d_sol)<epsilon:
            xk += d_sol
            break
        if mkd >= -1e-12:
            break

        # Armijo
        alpha = 1
        iter_armijo = 0
        while iter_armijo < max_iterarmijo:
            iter_armijo += 1
            x_trial = xk + alpha * d_sol
            faux_trial = np.array([f_i(ti, yi, x_trial) for ti, yi in zip(t, y)])
            fxk_trial = np.sort(faux_trial)[q]
            if fxk_trial <= fxk + theta * alpha * mkd:
                break
            alpha *= 0.5

        xk = x_trial
        print(iteracion, fxk, mkd, iter_armijo)

    print("Solución final:", xk)
    return xk

# Carga de datos
data = np.loadtxt("data.txt")
t = data[:,0]
y = data[:,1]

ovoqn(t, y)
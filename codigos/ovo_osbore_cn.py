import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def model(t,x0,x1,x2,x3,x4):
    return x0 + (x1 * np.exp(-t * x3)) + (x2 * np.exp(-t * x4))

def f_i(t_i,y_i,x):
    return 0.5 * ((model(t_i,*x) - y_i) ** 2)

def grad_f_i(t_i, y_i, x, grad):
    diff = model(t_i, *x) - y_i
    grad[0] = diff * 1
    grad[1] = diff * np.exp(-t_i * x[3])
    grad[2] = diff * np.exp(-t_i * x[4])
    grad[3] = diff * -t_i * x[1] * np.exp(-t_i * x[3])
    grad[4] = diff * -t_i * x[2] * np.exp(-t_i * x[4])
    return grad[:]

def hess_f_i(t_i, y_i, x):
    grad = np.zeros(5)
    grad = grad_f_i(t_i, y_i, x, grad)
    H = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            H[i,j] = grad[i] * grad[j]
    return H

def mount_Idelta(fovo, faux, indices, delta, Idelta, types, m):
    k = 0
    for i in range(m):
        diff = abs(fovo - faux[i])
        if diff <= delta:
            Idelta[k] = indices[i]
            if diff < delta/2:
                types[k] = 'eq'
            else:
                types[k] = 'ineq'
            k += 1
    return k

def compute_Bkj(H, first_iter=False):
    if first_iter:
        return np.eye(H.shape[0])
    Hs = 0.5*(H + H.T)
    eigs = np.linalg.eigvalsh(Hs)
    lambda_min = np.min(eigs)
    ajuste = max(0, -lambda_min + 1e-8)
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
    epsilon = 1e-8
    delta = 1e-4
    theta = 0.5
    q = 35
    max_iter = 200
    max_iterarmijo = 100

    m = len(t)
    q = min(q, m-1)
    xk = np.array([0.5, 1.5, -1, 0.01, 0.02])
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)
    types  = np.empty(m, dtype=object)

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
            g = np.zeros(5)
            grad_f_i(t[ind], y[ind], xk, g)
            H = hess_f_i(t[ind], y[ind], xk)
            B_full = compute_Bkj(H, first_iter=(iteracion==1))
            Bkjs.append(B_full[:4,:4])
            grads.append(g[:4])
            constr_types.append(types[r])

        x0 = np.zeros(5)

        constraints = []
        for g, B, ctype in zip(grads, Bkjs, constr_types):
            constraints.append({
                'type': ctype,
                'fun': lambda var, g=g, B=B: constraint_fun(var, g, B),
                'jac': lambda var, g=g, B=B: constraint_jac(var, g, B)
            })

        res = minimize(lambda var: var[4], x0, method="SLSQP", constraints=constraints,
                       options={'ftol':1e-8, 'maxiter':100, 'disp':False})

        d_sol = res.x[:4]
        mkd = float(res.fun)

        if abs(mkd)<epsilon or np.linalg.norm(d_sol)<epsilon:
            xk[:4] += d_sol
            break
        if mkd >= -1e-12:
            break

        alpha = 1
        iter_armijo = 0
        while iter_armijo < max_iterarmijo:
            iter_armijo += 1
            d_full = np.zeros_like(xk)
            d_full[:4] = d_sol
            x_trial = xk + alpha * d_full
            faux_trial = np.array([f_i(ti, yi, x_trial) for ti, yi in zip(t, y)])
            fxk_trial = np.sort(faux_trial)[q]
            if fxk_trial <= fxk + theta * alpha * mkd:
                break
            alpha *= 0.5

        xk = x_trial
        print(iteracion, fxk, mkd, iter_armijo)

    print("Solución final:", xk)
    return xk

# ===========================
# Ejecutar y graficar
# ===========================
data = np.loadtxt("data_osborne1.txt")
t = data[:,0]
y = data[:,1]

xk_final = ovoqn(t, y)
y_pred = model(t, *xk_final)

plt.figure(figsize=(8,5))
plt.scatter(t, y, color="blue", label="Datos observados")
plt.plot(t, y_pred, color="red", linewidth=2, label="Modelo ajustado (OVOQN)")
plt.legend()
plt.show()
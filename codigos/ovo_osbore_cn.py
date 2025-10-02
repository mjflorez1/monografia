import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def model(t, x0, x1, x2, x3, x4):
    return x0 + (x1 * np.exp(-t * x3)) + (x2 * np.exp(-t * x4))

def f_i(ti, yi, x):
    return 0.5 * ((model(ti, *x) - yi)**2)

def grad_f_i(ti, yi, x, grad):
    diff = model(ti, *x) - yi
    grad[0] = diff
    grad[1] = diff * np.exp(-ti * x[3])
    grad[2] = diff * np.exp(-ti * x[4])
    grad[3] = diff * (-ti * x[1] * np.exp(-ti * x[3]))
    grad[4] = diff * (-ti * x[2] * np.exp(-ti * x[4]))
    return grad

def hess_f_i(ti, yi, x):
    grad = np.zeros(5)
    grad_f_i(ti, yi, x, grad)
    H = np.outer(grad, grad)
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
    d = var[:5]
    z = var[5]
    return z - (g.dot(d) + 0.5*d.dot(B.dot(d)))

def constraint_jac(var, g, B):
    d = var[:5]
    gradc = np.zeros(6)
    gradc[:5] = -(g + B.dot(d))
    gradc[5] = 1.0
    return gradc

def ovoqn(t, y):
    epsilon = 1e-9
    delta = 1e-3
    deltax = 0.5
    theta = 0.5
    q = 27
    max_iter = 200
    max_iterarmijo = 100

    m = len(t)
    q = min(q, m-1)
    xk = np.array([0.4, 1.2, -0.8, 0.05, 0.03])
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)
    types = np.empty(m, dtype=object)

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
            Bkjs.append(compute_Bkj(H, first_iter=(iteracion==1)))
            grads.append(g)
            constr_types.append(types[r])

        x0 = np.zeros(6)
        bounds = [(-deltax, deltax)] * 5 + [(None, 0.0)]

        constraints = []
        for g, B, ctype in zip(grads, Bkjs, constr_types):
            constraints.append({
                'type': ctype,
                'fun': lambda var, g=g, B=B: constraint_fun(var, g, B),
                'jac': lambda var, g=g, B=B: constraint_jac(var, g, B)
            })

        res = minimize(lambda var: var[5], x0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={'ftol':1e-8, 'maxiter':100, 'disp':False})

        d_sol = res.x[:5]
        mkd = float(res.fun)

        if abs(mkd)<epsilon or np.linalg.norm(d_sol)<epsilon:
            xk += d_sol
            break
        if mkd >= -1e-12:
            break

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
        print(iteracion, fxk_trial, mkd, iter_armijo)

    print("Solución final:", xk)
    return xk

def model_safe(t, x0, x1, x2, x3, x4):
    arg1 = np.clip(-t * x3, -700, 700)
    arg2 = np.clip(-t * x4, -700, 700)
    return x0 + (x1 * np.exp(arg1)) + (x2 * np.exp(arg2))

data = np.loadtxt("data_osborne1.txt")
t = data[:,0]
y = data[:,1]

xk_final = ovoqn(t, y)
y_pred = model_safe(t, *xk_final)

plt.scatter(t, y, color="blue", alpha=0.6, label="Datos observados")
plt.plot(t, y_pred, color="red", linewidth=2, label="Modelo ajustado OVOQN")
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("figuras/ovoqn_osborne.png", bbox_inches="tight", dpi=150)
plt.show()
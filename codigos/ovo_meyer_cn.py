import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tabulate import tabulate

def model(t_i, x0, x1, x2):
    return x0 * np.exp(x1 / (t_i + x2))

def f_i(t_i, y_i, x):
    return 0.5 * ((model(t_i,*x) - y_i) ** 2)

def grad_f_i(t_i, y_i, x, grad):
    x0, x1, x2 = x
    z = np.exp(x1 / (t_i + x2))
    diff = (x0 * z) - y_i
    grad[0] = diff * z
    grad[1] = diff * x0 * z / (t_i + x2)
    grad[2] = diff * (-x0 * x1 * z / ((t_i + x2)**2))
    return grad[:]

def hess_f_i(t_i,y_i,x):
    grad = np.zeros(3)
    grad_f_i(t_i, y_i, x, grad)
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
    epsilon = 1e-2
    delta = 1e5
    deltax = 50.0
    theta = 0.5
    q = 12
    max_iter = 200
    max_iterarmijo = 100

    m = len(t)
    q = min(q, m-1)
    xk = np.array([0.02, 4000.0, 250.0])
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
            H = hess_f_i(t[ind], y[ind], xk)
            Bkjs.append(compute_Bkj(H, first_iter=(iteracion==1)))
            grads.append(g)
            constr_types.append(types[r])

        x0 = np.zeros(4)
        bounds = [
            (-deltax, deltax),
            (-deltax, deltax),
            (-deltax, deltax),
            (None, 0.0)
        ]

        constraints = []
        for g, B, ctype in zip(grads, Bkjs, constr_types):
            constraints.append({
                'type': ctype,
                'fun': lambda var, g=g, B=B: constraint_fun(var, g, B),
                'jac': lambda var, g=g, B=B: constraint_jac(var, g, B)
            })

        res = minimize(lambda var: var[3], x0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={'ftol':1e-8, 'maxiter':100, 'disp':False})

        d_sol = res.x[:3]
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
        table.append([fxk, iteracion, iter_armijo, mkd, nconst, Idelta[:min(5, nconst)].tolist()])

    print(tabulate(table, headers=header, tablefmt="grid"))
    print("Solución final:", xk)
    return xk

data = np.loadtxt("data_meyer.txt")
t = data[:,0]
y = data[:,1]

xk_final = ovoqn(t, y)
y_pred = model(t, *xk_final)

plt.scatter(t, y, color="blue", alpha=0.6, label="Datos observados")
plt.plot(t, y_pred, color="red", linewidth=2, label="Modelo ajustado OVOQN")
plt.legend()
plt.savefig("figuras/ovoqn_meyer.png", bbox_inches="tight", dpi=150)
plt.show()
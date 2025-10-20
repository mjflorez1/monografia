import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tabulate import tabulate
import os

# ------------------- MODELO DE OSBORNE 1 -------------------
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
    return np.outer(grad, grad)

# ------------------- FUNCIONES AUXILIARES -------------------
def mount_Idelta(fovo, faux, indices, delta, Idelta, types, m):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            types[k] = 'ineq'
            k += 1
    return k

def compute_Bkj(H):
    Hs = 0.5 * (H + H.T)
    eigs = np.linalg.eigvalsh(Hs)
    lambda_min = np.min(eigs)
    ajuste = max(0, -lambda_min + 1e-6)
    B = Hs + ajuste * np.eye(Hs.shape[0])
    return 0.5 * (B + B.T)

def constraint_fun(var, g, B):
    d = var[:5]
    z = var[5]
    return z - (g.dot(d) + 0.5 * d.dot(B.dot(d)))

def constraint_jac(var, g, B):
    d = var[:5]
    gradc = np.zeros(6)
    gradc[:5] = -(g + B.dot(d))
    gradc[5] = 1.0
    return gradc

# ------------------- ALGORITMO OVOQN -------------------
def ovoqn(t, y):
    epsilon = 1e-11
    delta = 1e-4
    deltax = 1
    theta = 0.2
    q = 27
    max_iter = 200
    max_iterarmijo = 50

    m = len(t)
    xk = np.array([0.5, 1.5, -1.0, 0.01, 0.02])
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)
    types = np.empty(m, dtype=object)

    header = ["f(xk)", "Iter", "IterArmijo", "Mk(d)", "ncons", "Idelta"]
    table = []

    for iteracion in range(1, max_iter + 1):
        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)

        indices = np.argsort(faux)
        faux_sorted = np.sort(faux)
        fxk = faux_sorted[q]

        nconst = mount_Idelta(fxk, faux_sorted, indices, delta, Idelta, types, m)
        if nconst == 0:
            break

        grads, Bkjs, constr_types = [], [], []
        for r in range(nconst):
            ind = Idelta[r]
            g = np.zeros(5)
            grad_f_i(t[ind], y[ind], xk, g)
            H = hess_f_i(t[ind], y[ind], xk)
            grads.append(g)
            Bkjs.append(compute_Bkj(H))
            constr_types.append(types[r])

        x0 = np.zeros(6)
        bounds = [(-deltax, deltax)] * 5 + [(None, 0.0)]

        constraints = [{
            'type': ctype,
            'fun': lambda var, g=g, B=B: constraint_fun(var, g, B),
            'jac': lambda var, g=g, B=B: constraint_jac(var, g, B)
        } for g, B, ctype in zip(grads, Bkjs, constr_types)]

        res = minimize(lambda var: var[5], x0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={'ftol': 1e-9, 'maxiter': 30, 'disp': False})

        d_sol = res.x[:5]
        mkd = float(res.fun)
        if abs(mkd) < epsilon:
            xk += d_sol
            break

        alpha = 1
        for iter_armijo in range(1, max_iterarmijo + 1):
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

# ------------------- AJUSTE OLS (L-BFGS-B) -------------------
def objetivo(params, t, y):
    return 0.5 * np.sum((model(t, *params) - y)**2)

# ------------------- DATOS -------------------
data = np.loadtxt("data_osborne1.txt")
t = data[:, 0]
y = data[:, 1]
os.makedirs("figuras", exist_ok=True)

# OVOQN
xk_final = ovoqn(t, y)
y_ovo = model(t, *xk_final)

# L-BFGS-B
x0 = [0.5, 1.5, -1.0, 0.01, 0.02]
res = minimize(objetivo, x0, args=(t, y), method="L-BFGS-B")
y_ols = model(t, *res.x)

print("\n=== AJUSTE L-BFGS-B ===")
print(f"x = {res.x}")
print(f"Función objetivo final: {res.fun:.6e}")

# ------------------- GRAFICAR AMBOS AJUSTES -------------------
plt.scatter(t, y, color="gray", s=40, alpha=0.6, label="$Observaciones$")
plt.plot(t, y_ols, 'b-', lw=2, label="$OLS$")
plt.plot(t, y_ovo, 'g-', lw=2, label="$OVO$")
plt.legend(loc="best")
plt.savefig("figuras/comparacionqn_osborne1.pdf", bbox_inches="tight")
plt.show()
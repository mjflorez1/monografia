import numpy as np
from scipy.optimize import minimize
from tabulate import tabulate
import matplotlib.pyplot as plt
import os

# ---------------------------
# Modelo Gaussiano y derivadas
# Modelo: x0 * exp(-x1*(t - x2)^2 / 2)
# Parámetros: x = (A, beta, mu)
# ---------------------------
def model_gauss(t, A, beta, mu):
    return A * np.exp(-beta * (t - mu)**2 / 2.0)

def resid(ti, yi, x):
    return model_gauss(ti, *x) - yi

def f_i(ti, yi, x):
    r = resid(ti, yi, x)
    return 0.5 * r * r

def jac_model(ti, x):
    A, beta, mu = x
    e = np.exp(-beta * (ti - mu)**2 / 2.0)
    # partials of model wrt A, beta, mu
    dm_dA = e
    dm_dbeta = -0.5 * A * (ti - mu)**2 * e
    dm_dmu = A * beta * (ti - mu) * e
    return np.array([dm_dA, dm_dbeta, dm_dmu])

def hess_model(ti, x):
    A, beta, mu = x
    e = np.exp(-beta * (ti - mu)**2 / 2.0)
    tmu = ti - mu
    # second derivatives of model
    d2m_dA2 = 0.0
    d2m_dAdbeta = -0.5 * (tmu**2) * e
    d2m_dAdmu = beta * tmu * e
    d2m_dbeta2 = 0.25 * A * (tmu**4) * e
    d2m_dbetamumu = -A * (0.5 * (tmu**2) + beta * (tmu**3) * 0.0) * e  # simplify below
    # exact expressions:
    d2m_dbetamumu = 0.5 * A * (tmu**3) * e  # careful: we derive symmetric terms properly below
    d2m_dmu2 = A * beta * e * (1 - beta * (tmu**2))
    # build symmetric Hessian of model
    H = np.zeros((3,3))
    H[0,0] = d2m_dA2
    H[0,1] = d2m_dAdbeta
    H[1,0] = H[0,1]
    H[0,2] = d2m_dAdmu
    H[2,0] = H[0,2]
    H[1,1] = d2m_dbeta2
    # cross beta-mu: derivative of dm_dbeta wrt mu
    H[1,2] = A * ( (tmu**3) * 0.5 ) * e  * (-1)  # approximate consistent cross-term sign
    H[2,1] = H[1,2]
    H[2,2] = d2m_dmu2
    return H

# For stability we will compute Hessian of f_i as:
# H_f = jac_model jac_model^T + r * hess_model
def grad_f_i(ti, yi, x):
    r = resid(ti, yi, x)
    J = jac_model(ti, x)
    return r * J

def hess_f_i(ti, yi, x):
    r = resid(ti, yi, x)
    J = jac_model(ti, x)
    Hm = hess_model(ti, x)
    return np.outer(J, J) + r * Hm

# ---------------------------
# Helpers OVO-QN (adaptados para n=3)
# ---------------------------
def mount_Idelta(fovo, faux_sorted, indices, delta):
    close = np.where(np.abs(faux_sorted - fovo) <= delta)[0]
    return indices[close].astype(int).tolist()

def compute_Bkj(H):
    Hs = 0.5 * (H + H.T)
    eigs = np.linalg.eigvalsh(Hs)
    lambda_min = np.min(eigs)
    ajuste = max(0.0, -lambda_min + 1e-8)
    B = Hs + ajuste * np.eye(Hs.shape[0])
    return 0.5 * (B + B.T)

def constraint_fun(var, g, B):
    d = var[:3]
    z = var[3]
    return z - (g.dot(d) + 0.5 * d.dot(B.dot(d)))

def constraint_jac(var, g, B):
    d = var[:3]
    gradc = np.zeros(4)
    gradc[:3] = -(g + B.dot(d))
    gradc[3] = 1.0
    return gradc

# ---------------------------
# Algoritmo OVO Quasi-Newton adaptado (n=3)
# ---------------------------
def ovoqn_gauss(t, y, x0=None, params=None, q=None):
    if params is None:
        params = {
            'epsilon': 1e-9,
            'delta': 1e-2,
            'deltax': 1.0,
            'theta': 0.7,
            'max_iter': 200,
            'max_iterarmijo': 80
        }
    epsilon = params['epsilon']
    delta = params['delta']
    deltax = params['deltax']
    theta = params['theta']
    max_iter = params['max_iter']
    max_iterarmijo = params['max_iterarmijo']

    m = len(t)
    if x0 is None:
        xk = np.array([1.0, 1.0, 0.0], dtype=float)  # A, beta, mu initial guess
    else:
        xk = np.array(x0, dtype=float)

    if q is None:
        q = max(0, min(m-1, m//3))

    faux = np.zeros(m)
    header = ["f(xk)", "Iter", "IterArmijo", "Mk(d)", "ncons", "Idelta (up to 6)"]
    table = []

    for iteracion in range(1, max_iter+1):
        # evaluar f_i en xk
        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)

        indices = np.argsort(faux)
        faux_sorted = faux[indices]
        fxk = faux_sorted[q]

        Idelta = mount_Idelta(fxk, faux_sorted, indices, delta)
        nconst = len(Idelta)
        if nconst == 0:
            break

        grads = []
        Bkjs = []
        for ind in Idelta:
            g = grad_f_i(t[ind], y[ind], xk)
            Hf = hess_f_i(t[ind], y[ind], xk)
            # use Hf as Hessian, but ensure PSD via compute_Bkj
            Bkjs.append(compute_Bkj(Hf))
            grads.append(g)

        # subproblema: var = [d0,d1,d2,z]
        x0_var = np.zeros(4)
        bounds = []
        for i in range(3):
            lo = max(-10.0 - xk[i], -deltax)
            hi = min(10.0 - xk[i], deltax)
            bounds.append((lo, hi))
        bounds.append((None, 0.0))

        constraints = []
        for g, B in zip(grads, Bkjs):
            constraints.append({
                'type': 'ineq',
                'fun': lambda var, g=g, B=B: constraint_fun(var, g, B),
                'jac': lambda var, g=g, B=B: constraint_jac(var, g, B)
            })

        def obj_var(var):
            return var[3]
        def obj_jac(var):
            grad = np.zeros(4)
            grad[3] = 1.0
            return grad

        res = minimize(obj_var, x0_var, method="SLSQP", jac=obj_jac,
                       bounds=bounds, constraints=constraints,
                       options={'ftol':1e-9, 'maxiter':100})

        if not res.success:
            print(f"[WARNING] Subproblema no convergió en iter {iteracion}: {res.message}")
            break

        d_sol = res.x[:3].copy()
        mkd = float(res.fun)

        if abs(mkd) < epsilon and np.linalg.norm(d_sol) < 1e-12:
            xk = xk + d_sol
            break

        # Armijo backtracking
        alpha = 1.0
        iter_armijo = 0
        x_trial = xk + alpha * d_sol
        while iter_armijo < max_iterarmijo:
            iter_armijo += 1
            faux_trial = np.array([f_i(ti, yi, x_trial) for ti, yi in zip(t, y)])
            fxk_trial = np.sort(faux_trial)[q]
            if fxk_trial <= fxk + theta * alpha * mkd:
                break
            alpha *= 0.5
            x_trial = xk + alpha * d_sol

        xk = x_trial.copy()
        table.append([float(fxk), iteracion, iter_armijo, float(mkd), nconst, Idelta[:6]])

    print(tabulate(table, headers=header, tablefmt="grid", floatfmt=".6f"))
    print("\nSolución final xk =", np.round(xk, 8))
    return xk

DATAFILE = "data.txt"
if not os.path.isfile(DATAFILE):
    raise FileNotFoundError("Coloca 'data.txt' (dos columnas: t y) en el directorio actual.")

data = np.loadtxt(DATAFILE)
t = data[:,0]
y = data[:,1]

# parámetros iniciales y del algoritmo
x0 = [1.0, 0.5, 0.0]   # A, beta, mu (ajusta si sabes mejores valores)
params = {'epsilon':1e-9, 'delta':1e-2, 'deltax':1.0, 'theta':0.7, 'max_iter':200, 'max_iterarmijo':80}
q = 35  # p-ésimo orden; ajusta según tus datos

xk_final = ovoqn_gauss(t, y, x0=x0, params=params, q=q)
y_pred = model_gauss(t, *xk_final)

os.makedirs("figuras", exist_ok=True)
order = np.argsort(t)
plt.figure()
plt.scatter(t, y, label="Datos")
plt.plot(t[order], y_pred[order], label="Ajuste Gaussiano", linewidth=1.5)
plt.legend()
plt.xlabel("t")
plt.ylabel("y")
plt.title("Ajuste OVO Cuasi-Newton (modelo gaussiano)")
plt.savefig("figuras/ovo_cn_gauss.pdf", bbox_inches="tight")
plt.show()
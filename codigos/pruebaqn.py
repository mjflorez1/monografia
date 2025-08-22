import numpy as np
from scipy.optimize import minimize

# =====================================================
# Modelo cúbico
# =====================================================
def model(t, x):
    return x[0] + x[1]*t + x[2]*(t**2) + x[3]*(t**3)

def f_i(ti, yi, x):
    return 0.5 * (model(ti, x) - yi)**2

def grad_f_i(ti, yi, x):
    diff = model(ti, x) - yi
    return np.array([diff, diff*ti, diff*(ti**2), diff*(ti**3)], dtype=float)

def hess_f_i(ti):
    phi = np.array([1.0, ti, ti**2, ti**3], dtype=float)
    return np.outer(phi, phi)

# =====================================================
# Armijo
# =====================================================
def armijo(fovo, t, y, xk, d, mkd, q, alpha0=1.0, rho=0.5, c=1e-4, max_iter=30):
    alpha = alpha0
    fxk = fovo
    for i in range(max_iter):
        faux_trial = np.array([f_i(ti, yi, xk + alpha*d) for ti, yi in zip(t, y)])
        fxk_trial = np.sort(faux_trial)[q]
        if fxk_trial <= fxk + c*alpha*mkd:
            return alpha, i+1
        alpha *= rho
    return 0.0, max_iter

# =====================================================
# BFGS
# =====================================================
def bfgs_update(Bk, sk, yk):
    if np.dot(sk, yk) <= 1e-12:
        return Bk
    rho = 1.0 / np.dot(sk, yk)
    I = np.eye(len(sk))
    Vk = I - rho * np.outer(sk, yk)
    return Vk.T @ Bk @ Vk + rho * np.outer(sk, sk)

# =====================================================
# Subproblema cuadrático con SLSQP
# =====================================================
def solve_subproblem(grads, Bk, Delta):
    n = 4
    nv = n + 1  # d (4) + z
    def obj(var): return float(var[-1])
    def obj_jac(var):
        grad = np.zeros(nv); grad[-1] = 1.0
        return grad

    cons = []
    for g in grads:
        cons.append({
            'type': 'ineq',
            'fun': lambda var, g=g: var[-1] - (g @ var[:n] + 0.5 * var[:n] @ (Bk @ var[:n])),
            'jac': lambda var, g=g: np.concatenate([-g - Bk @ var[:n], [1.0]])
        })

    bounds = [(-Delta, Delta)]*n + [(None, 0.0)]
    x0 = np.zeros(nv); x0[-1] = -1e-3

    res = minimize(obj, x0, jac=obj_jac, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"ftol": 1e-9, "maxiter": 100})
    return res.x[:n], float(res.fun)

# =====================================================
# Algoritmo OVO quasi-Newton con SLSQP
# =====================================================
def ovo_qnewton_slsqp(t, y, Delta0=1.0, q=35, tol=1e-6, max_iter=200):
    m = len(t)
    q = min(q, m-1)

    # Inicialización
    xk = np.array([-1.0, -2.0, 1.0, -1.0], dtype=float)
    Bk = np.eye(4)
    Delta = Delta0

    for k in range(max_iter):
        # Evaluar todas las fi
        faux = np.array([f_i(ti, yi, xk) for ti, yi in zip(t, y)])
        indices = np.argsort(faux)
        fxk = np.sort(faux)[q]

        # Conjunto activo
        Idelta = indices[:q+1]
        grads = [grad_f_i(t[i], y[i], xk) for i in Idelta]

        # Resolver subproblema
        d, mkd = solve_subproblem(grads, Bk, Delta)

        if abs(mkd) < tol or np.linalg.norm(d) < tol:
            print(f"[SLEEK] Iter {k}: parada |mkd|={abs(mkd):.2e}, ||d||={np.linalg.norm(d):.2e}")
            break

        # Armijo
        alpha, nback = armijo(fxk, t, y, xk, d, mkd, q)
        x_next = xk + alpha*d

        # BFGS
        sk = x_next - xk
        yk = np.mean([grad_f_i(t[i], y[i], x_next) - grad_f_i(t[i], y[i], xk) for i in Idelta], axis=0)
        Bk = bfgs_update(Bk, sk, yk)

        xk = x_next
        print(f"[SLEEK] Iter {k}: fxk={fxk:.6f}, mkd={mkd:.3e}, alpha={alpha:.3f}, Armijo={nback}")

    print("Solución final (QN-SLSQP OVO):", xk)
    return xk

# =====================================================
# Ejemplo de uso
# =====================================================
data = np.loadtxt("data.txt")
t = data[:, 0]
y = data[:, 1]

ovo_qnewton_slsqp(t, y, Delta0=1.0, q=35, max_iter=200)
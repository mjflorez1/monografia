import numpy as np
from scipy.optimize import minimize

# Modelo cúbico
def model(t, x1, x2, x3, x4):
    return x1 + (x2 * t) + x3 * (t**2) + x4 * (t**3)

def f_i(t_i, y_i, x):
    return 0.5 * ((model(t_i, *x) - y_i)**2)

def grad_f_i(t_i, y_i, x, grad):
    diff = model(t_i, *x) - y_i
    grad[0] = diff
    grad[1] = diff * t_i
    grad[2] = diff * (t_i**2)
    grad[3] = diff * (t_i**3)
    return grad

def hess_f_i(ti):
    phi = np.array([1.0, ti, ti**2, ti**3], dtype=float)
    return np.outer(phi, phi)

def mount_Idelta(fovo, faux, indices, delta, Idelta, m):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

def compute_Bkj(H, epsilon=1e-8, reg=1e-12):
    Hs = 0.5*(H + H.T)
    eigs = np.linalg.eigvalsh(Hs)
    lambda_min = np.min(eigs)
    ajuste = max(0.0, -lambda_min + epsilon)
    B = Hs + ajuste * np.eye(Hs.shape[0])
    B += reg * np.eye(Hs.shape[0])
    return 0.5*(B + B.T)

# Algoritmo quasi-Newton modificado para exportar
def ovo_qnewton_slsqp(t, y, p):
    epsilon = 1e-8
    delta   = 1e-3
    deltax  = 1.0
    theta   = 0.5
    max_iter = 500
    max_iter_armijo = 50

    m = len(t)
    q = p  # usamos p directamente como orden OVO
    q = min(q, m-1)

    # Solución inicial
    xk = np.array([-1.0, -2.0, 1.0, -1.0], dtype=float)
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)

    iter_count = 1
    while iter_count <= max_iter:
        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)

        indices = np.argsort(faux)
        faux_sorted = np.sort(faux)
        fxk = faux_sorted[q]

        nconst = mount_Idelta(fxk, faux_sorted, indices, delta, Idelta, m)
        if nconst == 0:
            break

        grads = []
        Bkjs = []
        for r in range(nconst):
            ind = Idelta[r]
            g = np.zeros(4)
            grad_f_i(t[ind], y[ind], xk, g)
            H = hess_f_i(t[ind])
            Bkj = compute_Bkj(H)
            grads.append(g.copy())
            Bkjs.append(Bkj.copy())

        nv = 5
        def obj(var): return float(var[-1])
        def obj_jac(var):
            grad_obj = np.zeros(nv, dtype=float)
            grad_obj[-1] = 1.0
            return grad_obj

        cons = []
        for j in range(nconst):
            g = grads[j]
            B = Bkjs[j]
            def make_constr(g_local, B_local):
                def constr(var):
                    d = var[:4]
                    z = var[4]
                    return float(z - (g_local.dot(d) + 0.5 * d.dot(B_local.dot(d))))
                def jac(var):
                    d = var[:4]
                    gradc = np.zeros(nv, dtype=float)
                    gradc[:4] = -g_local - B_local.dot(d)
                    gradc[4] = 1.0
                    return gradc
                return {'type': 'ineq', 'fun': constr, 'jac': jac}
            cons.append(make_constr(g, B))

        # Restricciones de caja
        bounds = []
        for i in range(4):
            lb = max(-10.0 - xk[i], -deltax)
            ub = min( 10.0 - xk[i],  deltax)
            bounds.append((lb, ub))
        bounds.append((None, 0.0))

        x0 = np.zeros(nv, dtype=float)
        x0[4] = -0.1

        res = minimize(obj, x0, method='SLSQP', jac=obj_jac,
                       bounds=bounds, constraints=cons,
                       options={'ftol':1e-9, 'maxiter':200})

        d_sol = res.x[:4]
        mkd = float(res.fun)

        if abs(mkd) < epsilon or np.linalg.norm(d_sol) < epsilon:
            xk = xk + d_sol
            break
        if mkd >= -1e-12:
            break

        alpha = 1.0
        iter_armijo = 0
        while iter_armijo < max_iter_armijo:
            iter_armijo += 1
            x_trial = xk + alpha * d_sol
            faux_trial = np.array([f_i(ti, yi, x_trial) for ti, yi in zip(t, y)])
            fxk_trial = np.sort(faux_trial)[q]
            if fxk_trial <= fxk + theta * alpha * mkd:
                break
            alpha *= 0.5

        xk = x_trial.copy()
        iter_count += 1

    # Calcular valor final fobj
    faux = np.array([f_i(ti, yi, xk) for ti, yi in zip(t, y)])
    fobj = np.sort(faux)[q]

    return p, xk[0], xk[1], xk[2], xk[3], fobj, iter_count

# ---- MAIN: leer datos y exportar resultados ----
data = np.loadtxt("data.txt")
t = data[:, 0]
y = data[:, 1]

resultados = [ovo_qnewton_slsqp(t, y, p) for p in range(20, 44)]

with open("qnovo.txt", "w") as f:
    f.write(f"{'p':>2} | {'x1':>15} {'x2':>15} {'x3':>15} {'x4':>15} | {'fobj':>15} | {'iters':>5}\n")
    for p, x1, x2, x3, x4, fobj, iters in resultados:
        f.write(f"{p:2d} | {x1:15.6f} {x2:15.6f} {x3:15.6f} {x4:15.6f} | {fobj:15.6f} | {iters:5d}\n")
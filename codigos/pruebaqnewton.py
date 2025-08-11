# qnewton_slsqp.py
import numpy as np
from scipy.optimize import minimize

# -------------------------
# Modelo y utilidades
# -------------------------
def model(t, x1, x2, x3, x4):
    return x1 + (x2 * t) + x3 * (t**2) + x4 * (t**3)

def f_i(t_i, y_i, x):
    return 0.5 * ((model(t_i, *x) - y_i)**2)

def grad_f_i(t_i, y_i, x, grad):
    diff = model(t_i, *x) - y_i
    grad[0] = diff * 1.0
    grad[1] = diff * t_i
    grad[2] = diff * (t_i**2)
    grad[3] = diff * (t_i**3)
    return grad[:]

# CORRECCIÓN: Hessiana correcta (phi phi^T)
def hess_f_i(ti):
    phi = np.array([1.0, ti, ti**2, ti**3], dtype=float)
    return np.outer(phi, phi)

# mount_Idelta idéntica a la tuya (usa m pasado)
def mount_Idelta(fovo, faux, indices, delta, Idelta, m):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

# Ajuste de autovalores para B_kj (garantiza PSD y agrega pequeño reg)
def compute_Bkj(H, epsilon=1e-8, reg=1e-12):
    Hs = 0.5*(H + H.T)
    eigs = np.linalg.eigvalsh(Hs)
    lambda_min = np.min(eigs)
    ajuste = max(0.0, -lambda_min + epsilon)
    B = Hs + ajuste * np.eye(Hs.shape[0])
    B += reg * np.eye(Hs.shape[0])
    return 0.5*(B + B.T)

# -------------------------
# Algoritmo quasi-Newton (subproblema resuelto por SLSQP)
# -------------------------
def ovo_qnewton_slsqp(t, y):
    # parámetros (igual que tus códigos previos)
    epsilon = 1e-8
    delta   = 1e-3
    deltax  = 1.0
    theta   = 0.5
    n = 5            # 4 parámetros + 1 variable artificial z
    q = 35
    max_iter = 500
    max_iter_armijo = 50

    m = len(t)
    q = min(q, m-1)

    # solución inicial (solo parámetros xk de dimensión 4)
    xk = np.array([-1.0, -2.0, 1.0, -1.0], dtype=float)

    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)

    iter_k = 1
    while iter_k <= max_iter:
        # evaluar f_i en xk
        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)

        indices = np.argsort(faux)
        faux_sorted = np.sort(faux)
        fxk = faux_sorted[q]

        # construir I_delta
        nconst = mount_Idelta(fxk, faux_sorted, indices, delta, Idelta, m)
        if nconst == 0:
            print("I_delta vacío en iter", iter_k)
            break

        # preparar gradientes y B_kj
        grads = []
        Bkjs = []
        rhs_list = []
        for r in range(nconst):
            ind = Idelta[r]
            g = np.zeros(4)
            grad_f_i(t[ind], y[ind], xk, g)
            H = hess_f_i(t[ind])
            Bkj = compute_Bkj(H, epsilon=1e-8, reg=1e-12)
            grads.append(g.copy())
            Bkjs.append(Bkj.copy())
            rhs_list.append(fxk - f_i(t[ind], y[ind], xk))

        # -------------------------
        # Subproblema (variables: d (4), z (1)) -> vector var = [d0,d1,d2,d3,z]
        # Minimize z  subject to: g_j^T d + 0.5 d^T Bkj d - z <= 0  (for j in I_delta)
        # and bounds: -deltax <= d_i <= deltax, z <= 0
        # -------------------------
        nv = 5
        def obj(var):
            # var[-1] is z
            return float(var[-1])
        def obj_jac(var):
            # gradient: d/d(d_i) = 0; d/dz = 1
            grad_obj = np.zeros(nv, dtype=float)
            grad_obj[-1] = 1.0
            return grad_obj

        # constraints list for SLSQP: must return >= 0
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
                    # derivative wrt d: -g - B d   ; wrt z: 1
                    gradc = np.zeros(nv, dtype=float)
                    gradc[:4] = -g_local - B_local.dot(d)
                    gradc[4] = 1.0
                    return gradc
                return {'type': 'ineq', 'fun': constr, 'jac': jac}
            cons.append(make_constr(g, B))

        # bounds for variables: d_i in [lb, ub] with respect to delta and box around xk (same convención que usas)
        bounds = []
        for i in range(4):
            lb = max(-10.0 - xk[i], -deltax)
            ub = min( 10.0 - xk[i],  deltax)
            bounds.append((lb, ub))
        bounds.append((None, 0.0))  # z <= 0

        # initial guess for subproblem
        x0 = np.zeros(nv, dtype=float)
        x0[4] = -0.1  # z inicial (negativo)

        # resolver con SLSQP
        res = minimize(obj, x0, method='SLSQP', jac=obj_jac,
                       bounds=bounds, constraints=cons,
                       options={'ftol':1e-9, 'maxiter':200})

        if (not res.success):
            # informa y salir o intentar un fallback
            print(f"Subproblema SLSQP no convergió en iter {iter_k}: {res.message}")
            # podrías intentar aumentar maxiter o usar otro x0; por ahora salimos
            break

        sol = res.x
        d_sol = sol[:4]
        z_sol = sol[4]
        mkd = float(res.fun)  # valor mínimo z

        print(f"Iter {iter_k:3d}: fxk={fxk:.6f}, mkd={mkd:.6e}, ||d||={np.linalg.norm(d_sol):.3e}")

        # criterio de parada (si no hay descenso significativo)
        if abs(mkd) < epsilon or np.linalg.norm(d_sol) < epsilon:
            print("Convergencia alcanzada (mkd pequeño o paso pequeño).")
            xk = xk + d_sol
            break

        if mkd >= -1e-12:
            print("Dirección no descendente (mkd >= 0). Parando.")
            break

        # Armijo (misma condición que tenías)
        alpha = 1.0
        iter_armijo = 0
        accepted = False
        while iter_armijo < max_iter_armijo:
            iter_armijo += 1
            x_trial = xk + alpha * d_sol
            # evaluar f ordenado en trial
            faux_trial = np.array([f_i(ti, yi, x_trial) for ti, yi in zip(t, y)])
            fxk_trial = np.sort(faux_trial)[q]
            if fxk_trial <= fxk + theta * alpha * mkd:
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            print("Armijo fallo en iter", iter_k)
            break

        # aceptar punto
        xk = x_trial.copy()
        iter_k += 1

    print("Sol final (parametros):", xk)
    return xk

# -------------------------
# cargar datos y ejecutar (sin if __main__)
# -------------------------
data = np.loadtxt("data.txt")
t = data[:,0]
y = data[:,1]

ovo_qnewton_slsqp(t, y)
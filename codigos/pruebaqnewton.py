import numpy as np
from quadprog import solve_qp

# ---------- modelo y utilidades (idénticas a tus funciones) ----------
def model(t, x1, x2, x3, x4):
    return x1 + x2*t + x3*(t**2) + x4*(t**3)

def f_i(ti, yi, x):
    return 0.5 * (model(ti, x[0], x[1], x[2], x[3]) - yi)**2

def grad_f_i(ti, yi, x, grad):
    diff = model(ti, x[0], x[1], x[2], x[3]) - yi
    grad[0] = diff * 1.0
    grad[1] = diff * ti
    grad[2] = diff * ti**2
    grad[3] = diff * ti**3
    return grad

# Montar I_delta (misma interfaz que tu versión original)
def mount_Idelta(fovo, faux, indices, delta, Idelta):
    k = 0
    m = len(faux)
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

# Actualización BFGS (con protección ante fallo de curvatura)
def bfgs_update(Bk, sk, yk, tol=1e-12):
    syk = np.dot(sk, yk)
    if syk <= tol:
        # no hacemos update si no se cumple la condición de curvatura
        return Bk
    Bsk = Bk.dot(sk)
    skBsk = np.dot(sk, Bsk)
    term1 = np.outer(yk, yk) / syk
    term2 = np.outer(Bsk, Bsk) / skBsk if skBsk > tol else 0.0
    return Bk + term1 - term2

# ---------- método quasi-Newton (estructura y nombres como el Cauchy) ----------
def ovo_qnewton_quadprog(t, y):
    # parámetros
    epsilon = 1e-8    # tolerancia / regularizador
    delta   = 1e-3
    deltax  = 1.0
    theta   = 0.5
    n = 5            # 4 parámetros + 1 variable artificial z
    q = 35
    max_iter = 1000
    max_iter_armijo = 50

    m = len(t)
    q = min(q, m-1)  # seguridad: q no mayor que m-1

    # solución inicial: xk = [x1,x2,x3,x4,z]
    xk = np.array([-1.0, -2.0, 1.0, -1.0, -1.0], dtype=float)

    # vectores auxiliares
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)

    # inicializar Bk (matriz 4x4 para d)
    n_params = 4
    Bk = np.eye(n_params)

    iter = 1
    while iter <= max_iter:
        # 1) evaluar todas las f_i en xk (solo con los 4 parametros, z no entra en f_i)
        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk[:4])

        # ordenar y obtener f_q
        indices = np.argsort(faux)
        faux_sorted = np.sort(faux)
        fxk = faux_sorted[q]

        # construir I_delta
        nconst = mount_Idelta(fxk, faux_sorted, indices, delta, Idelta)
        if nconst == 0:
            print("I_delta vacío en iter", iter)
            break

        # variables de decisión del subproblema: [d0..d3, z] tamaño nv = 5
        nv = n  # 4 + 1

        A_rows = []
        b_rows = []

        grad = np.zeros(n_params)
        # restricciones por cada i en I_delta: grad_i^T d - z <= fxk - f_i
        for kidx in range(nconst):
            ind = Idelta[kidx]
            grad = grad_f_i(t[ind], y[ind], xk[:4], grad)
            row = np.zeros(nv)
            row[:n_params] = grad
            row[-1] = -1.0
            rhs = fxk - f_i(t[ind], y[ind], xk[:4])
            A_rows.append(row)
            b_rows.append(rhs)

        # restricciones de caja sobre d: -deltax <= d_j <= deltax  (A x <= b)
        for j in range(n_params):
            row = np.zeros(nv); row[j] = 1.0
            A_rows.append(row); b_rows.append(deltax)
            row = np.zeros(nv); row[j] = -1.0
            A_rows.append(row); b_rows.append(deltax)

        # restricción para variable artificial z <= 0  ->  [0..0, 1] * x <= 0
        row = np.zeros(nv); row[-1] = 1.0
        A_rows.append(row); b_rows.append(0.0)

        # convertir A x <= b a la forma que espera quadprog:  Amat^T x >= b_qp
        Aineq = np.asarray(A_rows)   # shape (n_constraints, nv)
        bvec = np.asarray(b_rows)

        # construir matrices para quadprog
        # objetivo: min 1/2 d^T Bk d + z  -> variables [d (4), z (1)]
        Dmat = np.zeros((nv, nv), dtype=float)
        Dmat[:n_params, :n_params] = Bk + 1e-12*np.eye(n_params)  # pequeño regularizador
        c = np.zeros(nv, dtype=float)
        c[-1] = 1.0               # + z en la parte lineal
        dvec = -c                 # quadprog usa -d en la forma 1/2 x^T D x - d^T x

        # asegurar simetría de Dmat
        Dmat = (Dmat + Dmat.T) / 2.0

        # convertir desigualdades: (-Aineq) x >= -bvec
        Amat = (-Aineq).T
        b_qp = -bvec

        # garantizar que Dmat sea definida positiva para quadprog
        min_eig = np.min(np.real(np.linalg.eigvalsh(Dmat)))
        if min_eig <= 1e-12:
            Dmat += (abs(min_eig) + 1e-8) * np.eye(nv)

        # resolver QP con quadprog
        try:
            sol, f_val, xu, it, lag = solve_qp(Dmat, dvec, Amat, b_qp, meq=0)
        except Exception as e:
            # Algunas versiones devuelven menos valores; intentar forma reducida
            try:
                sol, f_val = solve_qp(Dmat, dvec, Amat, b_qp, meq=0)
            except Exception as e2:
                print("quadprog falló en iter", iter, ":", e2)
                break

        # sol es vector [d,z]
        d_sol = sol[:n_params].copy()
        z_sol = float(sol[-1])
        # dirección de búsqueda (d,z) en relación a xk
        dk = np.zeros(nv)
        dk[:n_params] = d_sol - xk[:n_params]
        dk[-1] = z_sol - xk[-1]

        # valor de la función objetivo del subproblema: 0.5 d^T Bk d + z
        obj_sol = 0.5 * d_sol.dot(Bk.dot(d_sol)) + z_sol
        obj_xk  = 0.5 * xk[:n_params].dot(Bk.dot(xk[:n_params])) + xk[-1]
        mkd = obj_sol - obj_xk

        print(f"{iter:3d}  fxk={fxk:.6f}   mkd={mkd:.6f}   ||d||={np.linalg.norm(dk[:n_params]):.6e}")

        # criterio de parada
        if abs(z_sol) < epsilon or np.linalg.norm(dk[:n_params]) < epsilon:
            print("Convergencia alcanzada (criterio en z o en ||d||).")
            xk[:n_params] = d_sol
            xk[-1] = z_sol
            break

        # Armijo (búsqueda de paso)
        alpha = 1.0
        iter_armijo = 0
        accepted = False
        while iter_armijo < max_iter_armijo:
            iter_armijo += 1
            xktrial = xk.copy()
            xktrial[:n_params] = xk[:n_params] + alpha * dk[:n_params]
            # en la convención que usamos en la otra implementación, fijamos z_trial = 0
            xktrial[-1] = 0.0

            # recalcular f ordenado en trial
            faux_trial = np.zeros(m)
            for i in range(m):
                faux_trial[i] = f_i(t[i], y[i], xktrial[:4])

            fxktrial = np.sort(faux_trial)[q]

            # condición Armijo: f(x+alpha d) <= f(x) + theta * alpha * mkd
            if fxktrial <= fxk + theta * alpha * mkd:
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            print("Búsqueda lineal falló en iter", iter)
            break

        # actualizar xk con el punto aceptado
        xk_new = xktrial.copy()

        # identificación del índice activo ip (el que ocupa la posición q en la ordenación)
        ip = indices[q]

        # gradientes del f_ip en xk y en xk_new (solo parámetros)
        gk = np.zeros(n_params)
        gp = np.zeros(n_params)
        grad_f_i(t[ip], y[ip], xk[:4], gk)
        grad_f_i(t[ip], y[ip], xk_new[:4], gp)

        # sk, yk para BFGS (sk = x_{k+1} - x_k, yk = grad_{ip}(x_{k+1}) - grad_{ip}(x_k))
        sk = xk_new[:n_params] - xk[:n_params]
        yk = gp - gk

        # actualizar Bk con BFGS (proteger curvatura)
        Bk = bfgs_update(Bk, sk, yk)

        # aceptar nuevo punto
        xk = xk_new.copy()
        iter += 1

    # imprimir y devolver resultado final (solo parámetros)
    print("Solución final (parámetros):")
    print(f"x1 = {xk[0]:.6f}")
    print(f"x2 = {xk[1]:.6f}")
    print(f"x3 = {xk[2]:.6f}")
    print(f"x4 = {xk[3]:.6f}")
    return xk[:4]

# ---------- cargar datos y ejecutar (sin guardia __main__) ----------
data = np.loadtxt("data.txt")
t = data[:,0]
y = data[:,1]

ovo_qnewton_quadprog(t, y)
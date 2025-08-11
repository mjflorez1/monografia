import numpy as np
from qpsolvers import solve_qp

# ========================
# Modelo cúbico
# ========================
def model(t, x1, x2, x3, x4):
    return x1 + (x2 * t) + x3 * (t ** 2) + x4 * (t ** 3)

# ========================
# Función de error cuadrático
# ========================
def f_i(ti, yi, x):
    return 0.5 * (model(ti, *x) - yi) ** 2

# ========================
# Gradiente de f_i
# ========================
def grad_f_i(ti, yi, x, grad):
    diff = model(ti, *x) - yi
    grad[0] = diff * 1
    grad[1] = diff * ti
    grad[2] = diff * ti**2
    grad[3] = diff * ti**3
    return grad[:]

# ========================
# Hessiana de f_i
# ========================
def hess_f_i(ti, H):
    H[0, 0] = 1
    H[0, 1] = H[1, 0] = ti
    H[0, 2] = H[2, 0] = ti**2
    H[0, 3] = H[3, 0] = ti**3
    H[1, 1] = ti**2
    H[1, 2] = H[2, 1] = ti**3
    H[1, 3] = H[3, 1] = ti**4
    H[2, 2] = ti**4
    H[2, 3] = H[3, 2] = ti**5
    H[3, 3] = ti**6
    return H

# ========================
# Construir I_delta
# ========================
def mount_Idelta(fovo, faux, indices, delta, Idelta):
    k = 0
    for i in range(len(faux)):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

# ========================
# Ajustar autovalores para B_kj
# ========================
def compute_Bkj(H, epsilon):
    eigvals = np.linalg.eigvalsh(H)
    lambda_min = np.min(eigvals)
    ajuste = max(0, -lambda_min + epsilon)
    return H + ajuste * np.eye(H.shape[0])

# ========================
# Algoritmo Quasi-Newton OVO
# ========================
def ovo_qnewton_algorithm(t, y):

    # Parámetros
    epsilon = 1e-8
    delta   = 1e-3
    deltax  = 1.0
    theta   = 0.5
    n       = 5
    q       = 35
    max_iter = 1000
    max_iter_armijo = 100
    iter    = 1

    m = len(t)
    xk = np.array([-1.0, -2.0, 1.0, -1.0])  # sin variable artificial

    # Auxiliares
    xktrial = np.zeros(4)
    faux    = np.zeros(m)
    Idelta  = np.zeros(m, dtype=int)

    # Vector c (minimiza variable artificial z)
    c = np.zeros(n)
    c[-1] = 1

    while iter <= max_iter:
        iter_armijo = 0

        # Restricciones de caja en forma Gx <= h
        box_G = []
        box_h = []
        for i in range(4):
            lb = max(-10 - xk[i], -deltax)
            ub = min( 10 - xk[i],  deltax)
            gi_lb = np.zeros(n); gi_lb[i] = -1; box_G.append(gi_lb); box_h.append(-lb)
            gi_ub = np.zeros(n); gi_ub[i] =  1; box_G.append(gi_ub); box_h.append( ub)
        # variable artificial ≤ 0
        gi_z  = np.zeros(n); gi_z[-1] = 1; box_G.append(gi_z); box_h.append(0.0)

        # Evaluar funciones de error
        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)

        indices = np.argsort(faux)
        faux = np.sort(faux)
        fxk = faux[q]

        # Construir I_delta
        nconst = mount_Idelta(fxk, faux, indices, delta, Idelta)

        # Construir B_k como promedio de B_kj
        Bk = np.zeros((4, 4))
        grad_list = []
        for i in range(nconst):
            ind = Idelta[i]
            grad = np.zeros(4)
            grad_f_i(t[ind], y[ind], xk, grad)
            grad_list.append(grad)

            Htemp = np.zeros((4, 4))
            hess_f_i(t[ind], Htemp)
            Bkj = compute_Bkj(Htemp, epsilon)
            Bk += Bkj
        Bk /= nconst

        # Construir matrices del QP
        P = np.zeros((n, n))
        P[:4, :4] = Bk
        q_vec = c.copy()

        G = np.array(box_G)
        h = np.array(box_h)

        # Agregar restricciones de I_delta: g_j^T d - z <= fxk - f_j
        for i in range(nconst):
            ind = Idelta[i]
            grad = grad_list[i]
            fval = f_i(t[ind], y[ind], xk)
            gi = np.zeros(n)
            gi[:4] = grad
            gi[-1] = -1
            G = np.vstack([G, gi])
            h = np.hstack([h, fxk - fval])

        # Resolver QP
        sol = solve_qp(P, q_vec, G, h, solver="quadprog")
        dk = sol[:4]
        mkd = sol[-1]

        print(f"Iter {iter}: fxk={fxk:.6f}, mkd={mkd:.6e}")

        # Criterio de parada
        if abs(mkd) < epsilon:
            break

        # Búsqueda lineal tipo Armijo
        alpha = 1.0
        while iter_armijo <= max_iter_armijo:
            iter_armijo += 1
            xktrial = xk + alpha * dk
            for i in range(m):
                faux[i] = f_i(t[i], y[i], xktrial)
            faux = np.sort(faux)
            fxktrial = faux[q]
            if fxktrial < fxk + theta * alpha * mkd:
                break
            alpha *= 0.5

        xk = xktrial.copy()
        iter += 1

    print("x* =", xk)

# ========================
# Ejecutar
# ========================
data = np.loadtxt("data.txt")
t = data[:, 0]
y = data[:, 1]
ovo_qnewton_algorithm(t, y)
# Bibliotecas esenciales
import numpy as np
from scipy.optimize import minimize

# Definición del modelo cúbico
def model(t, x1, x2, x3, x4):
    return x1 + (x2 * t) + x3 * (t**2) + x4 * (t**3)

# Funciones de error cuadrático
def f_i(ti, yi, x):
    return 0.5 * ((model(ti, x[0], x[1], x[2], x[3]) - yi)**2)

# Gradientes de las funciones de error
def grad_f_i(ti, yi, x, grad):
    diff = model(ti, x[0], x[1], x[2], x[3]) - yi
    grad[0] = diff * 1
    grad[1] = diff * ti
    grad[2] = diff * ti**2
    grad[3] = diff * ti**3
    return grad[:]

# Hessiana de las funciones de error
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

# Montamos el conjunto de índices I_delta
def mount_Idelta(fovo, faux, indices, delta, Idelta):
    k = 0
    for i in range(len(faux)):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

# Computamos Bkj con ajuste de autovalores
def compute_Bkj(H, epsilon):
    eigvals = np.linalg.eigvalsh(H)
    lambda_min = np.min(eigvals)
    ajuste = max(0, -lambda_min + epsilon)
    Bkj = H + ajuste * np.eye(H.shape[0])
    return Bkj

def ovo_newton(t, y):
    # Parámetros algorítmicos
    epsilon = 1e-8
    delta = 1e-3
    deltax = 1.0
    theta = 0.5
    n = 5  # 4 parámetros + 1 variable artificial
    q = min(35, len(t)-1)
    max_iter = 1000
    max_iter_armijo = 100
    iter = 1

    # Solución inicial
    xk = np.array([-1.0, -2.0, 1.0, -1.0, -1.0])  # Asegúrate de que la variable artificial sea negativa

    # Vectores auxiliares
    xktrial = np.zeros(n)
    faux = np.zeros(len(t))
    Idelta = np.zeros(len(t), dtype=int)

    # Vector c para minimizar variable artificial
    c = np.zeros(n)
    c[-1] = 1  # Minimizar la variable artificial

    while iter <= max_iter:
        iter_armijo = 0

        # Restricciones de caja para cada variable
        bounds = [
            (max(-10.0 - xk[0], -deltax), min(10.0 - xk[0], deltax)),
            (max(-10.0 - xk[1], -deltax), min(10.0 - xk[1], deltax)),
            (max(-10.0 - xk[2], -deltax), min(10.0 - xk[2], deltax)),
            (max(-10.0 - xk[3], -deltax), min(10.0 - xk[3], deltax)),
            (None, 0.0)  # variable artificial ≤ 0
        ]

        # Evaluar funciones de error
        for i in range(len(t)):
            faux[i] = f_i(t[i], y[i], xk)

        # Ordenar errores y sus índices
        indices = np.argsort(faux)
        faux = np.sort(faux)

        # Valor objetivo ordenado de orden q
        q_index = min(q, len(faux)-1)
        fxk = faux[q_index]

        # Construir conjunto I_delta
        nconst = mount_Idelta(fxk, faux, indices, delta, Idelta)

        if nconst == 0:
            print("Conjunto I_delta vacío")
            break

        # Construir matriz A y vector b para restricciones A x ≤ b
        A = np.zeros((nconst, n))
        b_vec = np.zeros(nconst)

        grad = np.zeros(4)

        for i in range(nconst):
            ind = Idelta[i]
            grad_f_i(t[ind], y[ind], xk, grad)
            Htemp = np.zeros((4, 4))
            hess_f_i(t[ind], Htemp)
            Bkj = compute_Bkj(Htemp, epsilon)
            # Corregir la construcción de las restricciones
            f_val = f_i(t[ind], y[ind], xk)
            A[i, :-1] = grad
            A[i, -1] = -1  # Por variable artificial
            b_vec[i] = fxk - f_val  # lado derecho corregido

        # Restricciones para minimize: ineq constraint fun(x) >= 0 => queremos que b_vec - A @ x >=0
        ineq_cons = {
            'type': 'ineq',
            'fun': lambda x, A=A, b=b_vec: b - np.dot(A, x),
            'jac': lambda x, A=A: -A
        }

        # Función objetivo: minimizar la variable artificial, fun(x) = c^T x
        fun = lambda x: np.dot(c, x)
        jac_fun = lambda x: c

        res = minimize(fun, x0=xk, bounds=bounds, constraints=[ineq_cons], 
                      method='SLSQP', jac=jac_fun, options={'ftol': 1e-12})

        if not res.success:
            print("Error en la optimización:", res.message)
            break

        dk = res.x - xk  # Dirección de búsqueda
        mkd = res.fun - np.dot(c, xk)  # Cambio en la función objetivo

        print(f"Iter {iter}: fxk={fxk:.6f}, mkd={mkd:.6f}, iter_armijo={iter_armijo}")

        # Criterio de parada
        if abs(mkd) < epsilon or np.linalg.norm(dk) < epsilon:
            print("Convergencia alcanzada")
            break

        # Búsqueda lineal Armijo
        alpha = 1.0
        while iter_armijo <= max_iter_armijo:
            iter_armijo += 1
            xktrial[:4] = xk[:4] + alpha * dk[:4]
            xktrial[4] = 0.0

            for i in range(len(t)):
                faux[i] = f_i(t[i], y[i], xktrial)

            faux = np.sort(faux)
            fxktrial = faux[q_index]

            if fxktrial <= fxk + theta * alpha * mkd:
                break

            alpha *= 0.5

        if iter_armijo > max_iter_armijo:
            print("Búsqueda lineal falló")
            break

        xk = xktrial.copy()
        iter += 1

    print("Solución final:")
    print(f"x1 = {xk[0]:.6f}")
    print(f"x2 = {xk[1]:.6f}")
    print(f"x3 = {xk[2]:.6f}")
    print(f"x4 = {xk[3]:.6f}")
    return xk[:4]

# Cargar datos
data = np.loadtxt("data.txt")
t = data[:, 0]
y = data[:, 1]

# Ejecutar algoritmo
ovo_newton(t, y)
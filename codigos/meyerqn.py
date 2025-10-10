import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tabulate import tabulate

def model(t, x0, x1, x2):
    return x0 * np.exp(x1 / (t + x2))

def f_i(t_i, y_i, x):
    return 0.5 * ((model(t_i, *x) - y_i) ** 2)

def grad_f_i(t_i, y_i, x, grad):
    z = np.exp(x[1] / (t_i + x[2]))
    diff = (x[0] * z) - y_i
    grad[0] = diff * z
    grad[1] = diff * x[0] * z / (t_i + x[2])
    grad[2] = diff * (-x[0] * x[1] * z / ((t_i + x[2]) ** 2))
    return grad[:]

def hess_f_i(t_i, y_i, x, H):
    s = t_i + x[2]
    z = np.exp(x[1] / s)
    diff = (x[0] * z) - y_i

    J = np.zeros(3)
    J[0] = z
    J[1] = x[0] * z / s
    J[2] = -x[0] * x[1] * z / (s ** 2)

    H_r = np.zeros((3, 3))
    H_r[0, 1] = H_r[1, 0] = z / s
    H_r[0, 2] = H_r[2, 0] = -x[1] * z / (s ** 2)
    H_r[1, 1] = x[0] * z / (s ** 2)
    H_r[1, 2] = H_r[2, 1] = -x[0] * z * (s + x[1]) / (s ** 3)
    H_r[2, 2] = x[0] * z * ((2 * x[1]) / (s ** 3) + (x[1] ** 2) / (s ** 4))

    for i in range(3):
        for j in range(3):
            H[i, j] = J[i] * J[j] + diff * H_r[i, j]
    return H[:]

def mount_Idelta(fovo, faux_sorted, indices, delta, Idelta, types, m):
    k = 0
    for i in range(m):
        diff = abs(fovo - faux_sorted[i])
        if diff <= delta:
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
    d = var[:3]  # AHORA d tiene 3 componentes
    z = var[3]   # z es el cuarto componente
    # Queremos: g·d + 0.5·dᵀBd - z <= 0
    return g.dot(d) + 0.5 * d.dot(B.dot(d)) - z

def constraint_jac(var, g, B):
    d = var[:3]
    gradc = np.zeros(4)  # ahora 4 componentes
    gradc[:3] = g + B.dot(d)
    gradc[3] = -1.0
    return gradc

def ovoqn(t, y):
    epsilon = 1e-9
    delta = 3e-3
    deltax = 1.2
    theta = 0.3
    q = min(12, len(t) - 1)  # evitar índice fuera de rango
    max_iter = 50
    max_iterarmijo = 20

    m = len(t)
    xk = np.array([0.02, 4000.0, 250.0], dtype=float)
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
            g = np.zeros(3)  # solo 3 componentes para gradiente
            grad_f_i(t[ind], y[ind], xk, g)
            H = np.zeros((3, 3))
            hess_f_i(t[ind], y[ind], xk, H)  # ¡CORREGIDO: ahora con todos los argumentos!
            Bkjs.append(compute_Bkj(H))
            grads.append(g)
            constr_types.append(types[r])

        # Variable de optimización: [d0, d1, d2, z] → 4 dimensiones
        x0 = np.zeros(4)
        bounds = [
            (-deltax, deltax),  # d0
            (-deltax, deltax),  # d1
            (-deltax, deltax),  # d2
            (0.0, None)         # z ≥ 0
        ]

        constraints = []
        for g, B, ctype in zip(grads, Bkjs, constr_types):
            constraints.append({
                'type': ctype,
                'fun': lambda var, g=g, B=B: constraint_fun(var, g, B),
                'jac': lambda var, g=g, B=B: constraint_jac(var, g, B)
            })

        # Minimizar z (que es var[3])
        res = minimize(lambda var: var[3], x0, method="SLSQP",
                       bounds=bounds, constraints=constraints)

        if not res.success:
            print(f"Advertencia: optimización fallida en iteración {iteracion}")
            d_sol = np.zeros(3)
            mkd = 0.0
        else:
            d_sol = res.x[:3]
            mkd = float(res.fun)

        if abs(mkd) < epsilon:
            xk += d_sol
            break

        alpha = 1.0
        iter_armijo = 0
        x_trial = xk.copy()
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

# === Ejecución ===
try:
    data = np.loadtxt("data_meyer.txt")
    t = data[:, 0]
    y = data[:, 1]
except FileNotFoundError:
    # Datos sintéticos si no existe el archivo
    np.random.seed(42)
    t = np.linspace(300, 450, 16)
    true_x = [0.005, 4000, 250]
    y_clean = model(t, *true_x)
    y = y_clean + np.random.normal(0, 1e-5, size=t.shape)
    print("Usando datos sintéticos (data_meyer.txt no encontrado)")

xk_final = ovoqn(t, y)
y_pred = model(t, *xk_final)

plt.figure(figsize=(8, 5))
plt.scatter(t, y, color="blue", label="Datos")
plt.plot(t, y_pred, color="red", label="Ajuste")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("figuras/meyercn.png", bbox_inches="tight")
plt.show()
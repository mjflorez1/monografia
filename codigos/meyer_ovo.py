import numpy as np
from scipy.optimize import linprog

# Modelo
def model(t, x0, x1, x2):
    z = x1 / t + x2
    z = np.clip(z, -700, 700)
    return x0 * np.exp(z)

# Funciones de error cuadrático
def f_i(t_i, y_i, x):
    return 0.5 * ((model(t_i, *x) - y_i) ** 2)

def grad_f_i(t_i, y_i, x, grad):
    x0, x1, x2 = x
    z = x1 / t_i + x2
    z = np.clip(z, -700, 700)
    exp_term = np.exp(z)
    diff = x0 * exp_term - y_i
    grad[0] = diff * exp_term
    grad[1] = diff * x0 * exp_term / t_i
    grad[2] = diff * x0 * exp_term
    return grad[:]

# Montamos el conjunto de índices I_delta
def mount_Idelta(fovo, faux, indices, delta, Idelta, m):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

def ovo(t, y):
    epsilon = 1e-8
    delta = 1e-3
    theta = 0.5
    n = 4
    q = 10
    max_iter = 1000
    max_iter_armijo = 100
    iter = 1
    m = len(t)

    # Punto inicial fijo
    xk = np.array([0.02, 4000, 250])

    xktrial = np.zeros(n - 1)
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)

    c = np.zeros(n)
    c[-1] = 1

    while iter <= max_iter:
        iter_armijo = 0

        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)

        indices = np.argsort(faux)
        faux = np.sort(faux)

        fxk = faux[q]
        nconst = mount_Idelta(fxk, faux, indices, delta, Idelta, m)

        A = np.zeros((nconst, n))
        b = np.zeros(nconst)
        grad = np.zeros((nconst, n - 1))

        for i in range(nconst):
            ind = Idelta[i]
            grad_f_i(t[ind], y[ind], xk, grad[i, :])
            A[i,:-1] = grad[i, :]
            A[i,-1] = -1
            b[i] = f_i(t[ind], y[ind], xk) - fxk

        for i in range(nconst):
            norm = np.linalg.norm(A[i, :-1])
            if norm > 1e-12:
                A[i, :] /= norm
                b[i] /= norm

        res = linprog(c, A_ub=A, b_ub=b)

        dk = res.x
        mkd = dk[-1]

        if (abs(mkd) < epsilon):
            break

        alpha = 1
        while iter_armijo <= max_iter_armijo:
            xktrial = xk + (alpha * dk[:-1])
            for i in range(m):
                faux[i] = f_i(t[i], y[i], xktrial)
            faux = np.sort(faux)
            fxktrial = faux[q]
            if fxktrial < fxk + (theta * alpha * mkd):
                break
            alpha *= 0.5
            iter_armijo += 1

        if iter_armijo > max_iter_armijo:
            break

        print(iter,fxk,mkd,nconst,iter_armijo)
        xk = xktrial
        iter += 1

    print('Solución final:', xk)

# Ejecutar
data = np.loadtxt('meyer_data.txt')
t = data[:, 0]
y = data[:, 1]

ovo(t, y)
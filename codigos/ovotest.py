import numpy as np
from scipy.optimize import linprog

def model(t,x1,x2,x3,x4):
    return x1 + x2*t + x3*t**2 + x4*t**3

def f_i(t_i, y_i, x):
    return 0.5 * (model(t_i, *x) - y_i)**2

def grad_f_i(t_i, y_i, x, grad):
    diff = model(t_i, *x) - y_i
    grad[0] = diff
    grad[1] = diff * t_i
    grad[2] = diff * t_i**2
    grad[3] = diff * t_i**3
    return grad

def mount_Idelta(fovo, faux, indices, delta, Idelta):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

def ovo_algorithm(t, y, q):
    epsilon = 1e-8
    delta = 1e-3
    deltax = 1.0
    theta = 0.5
    n = 5
    max_iter = 1000
    max_iter_armijo = 100
    iter = 1

    xk = np.array([-1, -2, 1, -1])
    xktrial = np.zeros(n - 1)
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)
    c = np.zeros(n)
    c[-1] = 1

    while iter <= max_iter:
        iter_armijo = 0

        bounds = [
            (max(-10 - xk[0], -deltax), min(10 - xk[0], deltax)),
            (max(-10 - xk[1], -deltax), min(10 - xk[1], deltax)),
            (max(-10 - xk[2], -deltax), min(10 - xk[2], deltax)),
            (max(-10 - xk[3], -deltax), min(10 - xk[3], deltax)),
            (None, 0)
        ]

        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)

        indices = np.argsort(faux)
        faux = np.sort(faux)
        fxk = faux[q]
        nconst = mount_Idelta(fxk, faux, indices, delta, Idelta)

        A = np.zeros((nconst, n))
        b = np.zeros(nconst)
        grad = np.zeros((nconst, n - 1))

        for i in range(nconst):
            ind = Idelta[i]
            grad_f_i(t[ind], y[ind], xk, grad[i, :])
            A[i, :-1] = grad[i, :]
            A[i, -1] = -1

        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)

        dk = res.x
        mkd = dk[-1]

        if abs(mkd) < epsilon:
            break

        alpha = 1
        while iter_armijo <= max_iter_armijo:
            iter_armijo += 1
            xktrial = xk + alpha * dk[:-1]
            for i in range(m):
                faux[i] = f_i(t[i], y[i], xktrial)
            faux = np.sort(faux)
            fxktrial = faux[q]
            if fxktrial < fxk + theta * alpha * mkd:
                break
            alpha *= 0.5

        xk = xktrial
        iter += 1

    for i in range(m):
        faux[i] = f_i(t[i], y[i], xk)
    fxk = np.sort(faux)[q]
    return q, *xk, fxk, iter

data = np.loadtxt("data.txt")
t = data[:, 0]
y = data[:, 1]
m = len(t)

resultados = [ovo_algorithm(t, y, p) for p in range(20,44)]

with open("resultados_ovo.txt", "w") as f:
    f.write(f"{'p':>2} | {'x1':>15} {'x2':>15} {'x3':>15} {'x4':>15} | {'fobj':>15} | {'iters':>5}\n")
    for p, x1, x2, x3, x4, fobj, iters in resultados:
        f.write(f"{p:2d} | {x1:15.6f} {x2:15.6f} {x3:15.6f} {x4:15.6f} | {fobj:15.6f} | {iters:5d}\n")
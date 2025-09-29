import numpy as np
from scipy.optimize import linprog

def model(t,x0,x1,x2):
    return x0 * np.exp(x1 / (t + x2))

def f_i(t_i, y_i, x):
    return 0.5 * ((model(t_i,*x) - y_i) ** 2)

def grad_f_i(t_i,y_i,x,grad):
    x0, x1, x2 = x
    z = np.exp(x1 / (t_i + x2))
    diff = (x0 * z) - y_i
    grad[0] = z
    grad[1] = x0 * z / (t_i + x2)
    grad[2] = -x0 * x1 * z / ((t_i + x2)**2)
    return diff * grad[:]

def mount_Idelta(fovo, faux, indices, delta, Idelta, m):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

def ovo(t, y):
    epsilon = 1e-8
    delta = 1e+8
    theta = 0.2
    n = 4
    q = 14
    max_iter = 200
    max_iter_armijo = 100
    iter = 1
    alpha = 1.0  # Norma L1 permitida
    m = len(t)

    xk = np.array([0.02, 4000, 250])

    xktrial = np.zeros(n - 1)
    faux = np.zeros(m)
    Idelta = np.empty(m, dtype=int)

    while iter <= max_iter:
        iter_armijo = 0

        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)

        indices = np.argsort(faux)
        faux = np.sort(faux)

        fxk = faux[q]
        nconst = mount_Idelta(fxk, faux, indices, delta, Idelta, m)

        grad = np.zeros((nconst, n - 1))

        for i in range(nconst):
            ind = Idelta[i]
            grad_f_i(t[ind], y[ind], xk, grad[i, :])

        p = n - 1
        num_vars = 2*p + 1  # d+, d-, w
        c_lp = np.zeros(num_vars)
        c_lp[-1] = 1

        A_ub_lp = []
        b_ub_lp = []

        for i in range(nconst):
            row = np.zeros(num_vars)
            row[:p] = grad[i, :]         # +grad * d+
            row[p:2*p] = -grad[i, :]     # -grad * d-
            row[-1] = -1                 # -w
            A_ub_lp.append(row)
            b_ub_lp.append(0)

        row = np.zeros(num_vars)
        row[:p] = 1
        row[p:2*p] = 1
        A_ub_lp.append(row)
        b_ub_lp.append(alpha)

        A_ub_lp = np.array(A_ub_lp)
        b_ub_lp = np.array(b_ub_lp)

        bounds = [(0, None)] * (2*p) + [(None, None)]

        res = linprog(c_lp, A_ub=A_ub_lp, b_ub=b_ub_lp, bounds=bounds, method="highs")

        if res.status != 0:
            break

        d_plus = res.x[:p]
        d_minus = res.x[p:2*p]
        dk = d_plus - d_minus
        mkd = res.x[-1]

        if abs(mkd) < epsilon:
            break

        alpha_k = 1
        while iter_armijo <= max_iter_armijo:
            
            iter_armijo += 1
            
            xktrial = xk + (alpha_k * dk)
            
            for i in range(m):
                faux[i] = f_i(t[i], y[i], xktrial)
                
            faux = np.sort(faux)
            fxktrial = faux[q]
            
            if fxktrial < fxk + (theta * alpha_k * mkd):
                break
            
            alpha_k = 0.5 * alpha_k

        print(iter, fxk, mkd, nconst, iter_armijo)
        
        xk = xktrial
        iter += 1

    print('Solución final:', xk)

data = np.loadtxt('meyer_data.txt')
t = data[:, 0]
y = data[:, 1]
ovo(t, y)
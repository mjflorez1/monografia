import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize
from tabulate import tabulate
import os

# -------------------- MODELO GAUSSIANO --------------------
def model(t, x0, x1, x2):
    return x0 * np.exp((-x1 * ((t - x2)**2)) / 2)

def f_i(t_i, y_i, x):
    return 0.5 * ((model(t_i, *x) - y_i)**2)

def grad_f_i(t_i, y_i, x):
    x0, x1, x2 = x
    s = t_i - x2
    z = np.exp(-0.5 * x1 * s**2)
    diff = x0 * z - y_i
    grad = np.zeros(3)
    grad[0] = diff * z
    grad[1] = diff * (-0.5 * x0 * s**2 * z)
    grad[2] = diff * (x0 * x1 * s * z)
    return grad

def mount_Idelta(fovo, faux, indices, epsilon, m):
    Idelta = []
    for i in range(m):
        if abs(fovo - faux[indices[i]]) <= epsilon:
            Idelta.append(indices[i])
    return np.array(Idelta, dtype=int)

def ovo(t, y):
    stop = 1e-8
    epsilon = 1e-3
    delta = 10
    theta = 0.2
    n = 4
    m = len(t)
    q = 11
    max_iter = 100
    max_iter_armijo = 30
    xk = np.array([1.0, 2.0, 1.0])
    faux = np.zeros(m)
    header = ["Iter", "f(xk)", "IterArmijo", "Mk(d)", "ncons", "Idelta"]
    table = []
    c = np.zeros(n)
    c[-1] = 1

    for iter in range(1, max_iter + 1):
        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)
        indices = np.argsort(faux)
        faux_sorted = faux[indices]
        fxk = faux_sorted[q]
        Idelta = mount_Idelta(fxk, faux, indices, epsilon, m)
        nconst = len(Idelta)
        if nconst == 0:
            break
        A = np.zeros((nconst, n))
        b = np.zeros(nconst)
        for i in range(nconst):
            ind = Idelta[i]
            grad = grad_f_i(t[ind], y[ind], xk)
            A[i, :3] = grad
            A[i, 3] = -1
        bounds = [(-delta, delta)] * 3 + [(None, 0)]
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if not res.success:
            break
        dk = res.x
        mkd = dk[3]
        if abs(mkd) < stop:
            break
        alpha = 1.0
        iter_armijo = 0
        while iter_armijo < max_iter_armijo:
            iter_armijo += 1
            xktrial = xk + alpha * dk[:3]
            faux_trial = np.array([f_i(ti, yi, xktrial) for ti, yi in zip(t, y)])
            fxktrial = np.sort(faux_trial)[q]
            if fxktrial <= fxk + theta * alpha * mkd:
                break
            alpha *= 0.5
        xk = xk + alpha * dk[:3]
        table.append([iter, fxk, iter_armijo, mkd, nconst, Idelta[:5].tolist()])

    print(tabulate(table, headers=header, tablefmt="grid", floatfmt=".6e"))
    print(f'Solución final OVO: x0={xk[0]:.6f}, x1={xk[1]:.6f}, x2={xk[2]:.6f}')
    return xk

# -------------------- L-BFGS-B (OLS) --------------------
def gauss(beta, x):
    return beta[0] * np.exp(-((x - beta[1])**2) / (2 * beta[2]**2))

def objetivo(beta, t, y):
    return 0.5 * np.sum((y - gauss(beta, t))**2)

# -------------------- DATOS --------------------
data = np.loadtxt('data_gauss.txt')
t = data[:, 0]
y = data[:, 1]
os.makedirs("figuras", exist_ok=True)

# Ajuste OVO
xk_final = ovo(t, y)
tt = np.linspace(t.min(), t.max(), 400)
y_ovo = model(tt, *xk_final)

# Ajuste OLS
beta0 = [0.4, 0.0, 2.0]
bounds = [(0, 1.0), (-5, 5), (0.1, 5.0)]
res = minimize(objetivo, beta0, args=(t, y), method='L-BFGS-B', bounds=bounds)
y_ols = gauss(res.x, tt)

print("\n=== AJUSTE L-BFGS-B ===")
print(f"β1={res.x[0]:.6f}, β2={res.x[1]:.6f}, β3={res.x[2]:.6f}")
print(f"Función objetivo final: {res.fun:.6e}")

# -------------------- GRAFICAR AMBOS AJUSTES --------------------
plt.scatter(t, y, color="blue", s=40, alpha=0.6, label="$Observaciones$")
plt.plot(tt, y_ols, 'g-', lw=2, label="$OLS$")
plt.plot(tt, y_ovo, 'r-', lw=2, label="$OVO$")
plt.xlabel("$t$")
plt.ylabel("$y$")
plt.legend(loc="upper left")
plt.savefig("figuras/comparacion_gauss.pdf", bbox_inches="tight")
plt.show()
import numpy as np
from scipy.optimize import minimize, linprog

# Leer los datos
def load_data(filename="data.txt"):
    with open(filename, 'r') as f:
        lines = f.readlines()
    m = int(lines[0])
    t = []
    y = []
    for line in lines[1:]:
        ti, yi = map(float, line.strip().split())
        t.append(ti)
        y.append(yi)
    return np.array(t), np.array(y)

# Modelo cúbico y funciones f_i(x)
def model(t, x):
    return x[0] + x[1]*t + x[2]*t**2 + x[3]*t**3

def f_i(x, t_i, y_i):
    return 0.5 * (model(t_i, x) - y_i)**2

def grad_f_i(x, t_i, y_i):
    diff = model(t_i, x) - y_i
    return diff * np.array([1, t_i, t_i**2, t_i**3])

# Construcción de I_delta(x)
def compute_I_delta(x, t, y, q, delta):
    m = len(t)
    f_vals = np.array([f_i(x, t[j], y[j]) for j in range(m)])
    indices = np.argsort(f_vals)
    f_q = f_vals[indices[q-1]]
    I_delta = [j for j in range(m) if abs(f_vals[j] - f_q) <= delta]
    return I_delta, f_vals, indices

# Guardar restricciones
def save_constraints(x, t, y, I_delta, iter_num, filename="restricciones.txt"):
    with open(filename, "a") as f:
        for j in I_delta:
            grad = grad_f_i(x, t[j], y[j])
            restr = " + ".join([f"{grad[k]:.4f}*x{k+1}" for k in range(len(grad))])
            f.write(f"Iteración {iter_num} - Restricción {j}: {restr} <= w\n")
        
# Subproblema usando linprog (simplificación)
def solve_subproblem(xk, t, y, I_delta, delta=0.1):
    n = len(xk)
    A = []
    b = []
    for j in I_delta:
        grad = grad_f_i(xk, t[j], y[j])
        A.append(grad)
        b.append(0.0)
    A = np.array(A)
    m = len(I_delta)

    # LP: min w  s.t. g_j^T d <= w
    c = np.zeros(n + 1)
    c[-1] = 1  # minimize w

    A_ub = np.hstack([A, -np.ones((m, 1))])
    b_ub = np.zeros(m)

    bounds = [(-delta, delta)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if res.success:
        d = res.x[:-1]
        w = res.x[-1]
        return d, w
    else:
        raise RuntimeError("Subproblem did not converge")
# Algoritmo OVO simplificado
def ovo_algorithm(t, y, q, delta, x0=None, max_iter=50, tol=1e-3):
    n = 4
    if x0 is None:
        xk = np.array([-1.0, -2.0, 1.0, -1.0])
    else:
        xk = np.copy(x0)

    for iter in range(max_iter):
        I_delta, f_vals, indices = compute_I_delta(xk, t, y, q, delta)
        save_constraints(xk, t, y, I_delta, iter)
        d, w = solve_subproblem(xk, t, y, I_delta)

        xtrial = xk + d
        f_trial = np.array([f_i(xtrial, t[j], y[j]) for j in range(len(t))])
        fx_trial = np.sort(f_trial)[q-1]

        # Condición de terminación
        if np.linalg.norm(d) < tol:
            break
        xk = xtrial
    return xk, fx_trial
# Main
def main():
    t, y = load_data()
    q = 36
    delta = 0.01
    xsol, fval = ovo_algorithm(t, y, q, delta)
    print("Solución encontrada:", xsol)
    print("Valor de la función OVO:", fval)

if __name__ == "__main__":
    main()
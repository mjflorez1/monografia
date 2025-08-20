import numpy as np
from scipy.optimize import minimize

# =====================================================
# Función objetivo de prueba (puedes cambiarla)
# =====================================================
def f(x):
    return (x[0] - 2)**2 + (x[1] + 3)**2 + 0.5*(x[2] - 5)**2 + (x[3] - 1)**2

def grad_f(x):
    return np.array([
        2*(x[0] - 2),
        2*(x[1] + 3),
        (x[2] - 5),
        2*(x[3] - 1)
    ])

# =====================================================
# Búsqueda de línea (Armijo)
# =====================================================
def armijo(f, xk, d, grad, alpha0=1.0, rho=0.5, c=1e-4, max_iter=20):
    alpha = alpha0
    fxk = f(xk)
    for i in range(max_iter):
        if f(xk + alpha*d) <= fxk + c*alpha*np.dot(grad, d):
            return alpha, i+1
        alpha *= rho
    return 0.0, max_iter

# =====================================================
# Subproblema cuadrático con SLSQP (trust-region aproximado)
# =====================================================
def solve_subproblem(grad, Bk, Delta):
    n = len(grad)

    def model(d):
        return grad @ d + 0.5 * d @ (Bk @ d)

    def model_grad(d):
        return grad + Bk @ d

    cons = ({
        "type": "ineq",
        "fun": lambda d: Delta**2 - np.dot(d, d),   # ||d||^2 <= Delta^2
        "jac": lambda d: -2*d
    })

    res = minimize(model, np.zeros(n), jac=model_grad, constraints=cons,
                   method="SLSQP", options={"ftol": 1e-9, "disp": False, "maxiter": 100})
    return res.x

# =====================================================
# Actualización BFGS
# =====================================================
def bfgs_update(Bk, sk, yk):
    if np.dot(sk, yk) <= 1e-10:
        return Bk  # evitar división por cero
    rho = 1.0 / np.dot(sk, yk)
    I = np.eye(len(sk))
    Vk = I - rho * np.outer(sk, yk)
    return Vk.T @ Bk @ Vk + rho * np.outer(sk, sk)

# =====================================================
# Método quasi-Newton OVO con SLSQP
# =====================================================
def ovo_quasi_newton_slsqp(f, grad_f, x0, Delta0=1.0, tol=1e-6, max_iter=200):
    xk = np.array(x0, dtype=float)
    Bk = np.eye(len(x0))  # Hessiana inicial (identidad)
    Delta = Delta0

    for k in range(max_iter):
        gk = grad_f(xk)

        # Subproblema con SLSQP
        d = solve_subproblem(gk, Bk, Delta)
        mkd = gk @ d + 0.5 * d @ (Bk @ d)

        if abs(mkd) < tol or np.linalg.norm(d) < tol:
            print(f"[SLEEK] Iter {k}: parada |mkd|={abs(mkd):.2e}, ||d||={np.linalg.norm(d):.2e}")
            break

        # Armijo
        alpha, nback = armijo(f, xk, d, gk)

        x_next = xk + alpha * d
        sk = x_next - xk
        yk = grad_f(x_next) - gk

        Bk = bfgs_update(Bk, sk, yk)
        xk = x_next

        print(f"[SLEEK] Iter {k}: fxk={f(xk):.6f}, mkd={mkd:.3e}, alpha={alpha:.3f}, Armijo={nback}")

    return xk

# =====================================================
# Ejemplo de uso
# =====================================================
x0 = np.array([0.0, 0.0, 0.0, 0.0])
sol = ovo_quasi_newton_slsqp(f, grad_f, x0, Delta0=1.0, tol=1e-6, max_iter=200)
print("Solución final (QN-SLSQP SLEEK):", sol)
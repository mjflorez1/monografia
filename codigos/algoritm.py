import numpy as np

# Datos del ejemplo (Ejemplo 2.1 del artículo)
A = np.array([
    [2, 3, 5],
    [1, 2, 4]
])

b = np.array([5, 11])

# Parámetros del algoritmo
mu = 1.0
epsilon = 1e-6
max_iter = 100
epsilon_reg = 1e-8  # regularización para la Hessiana

# Función objetivo dual suavizada
def g(u):
    denom = 1 - A.T @ u
    if np.any(denom <= 0):
        return np.inf  # fuera del dominio
    return -b.T @ u - mu * np.sum(np.log(denom))

# Gradiente de g(u)
def grad_g(u):
    denom = 1 - A.T @ u
    return -b + mu * (A @ (1 / denom))

# Hessiana de g(u)
def hessian_g(u):
    denom = 1 - A.T @ u
    W = np.diag((1 / denom) ** 2)
    return mu * A @ W @ A.T

# Método de punto interior
def interior_point_method(use_pinv=False):
    u = np.array([0.1, 0.1])  # punto inicial válido

    for k in range(max_iter):
        grad = grad_g(u)
        hess = hessian_g(u)

        # Regularizar la Hessiana
        hess += epsilon_reg * np.eye(hess.shape[0])

        # Resolver H d = -grad
        if use_pinv:
            d = -np.linalg.pinv(hess) @ grad
        else:
            d = np.linalg.solve(hess, -grad)

        # Búsqueda de línea para mantener factibilidad
        alpha = 1.0
        while np.any(1 - A.T @ (u + alpha * d) <= 0):
            alpha *= 0.5
            if alpha < 1e-10:
                raise ValueError("No se pudo encontrar paso factible")

        # Actualización
        u = u + alpha * d

        # Criterio de parada
        if np.linalg.norm(grad) < epsilon:
            print(f"Convergencia en iteración {k}")
            break

    return u, g(u)

# Ejecutar con np.linalg.solve (más rápido)
u_sol, val_sol = interior_point_method(use_pinv=False)
print("🔧 Solución con solve():")
print("u* =", u_sol)
print("g(u*) =", val_sol)

# Ejecutar con np.linalg.pinv (alternativa si falla solve)
u_pinv, val_pinv = interior_point_method(use_pinv=True)
print("\n🔁 Solución con pinv():")
print("u* =", u_pinv)
print("g(u*) =", val_pinv)
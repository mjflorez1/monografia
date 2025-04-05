from scipy.optimize import minimize

# Función objetivo
def objetivo(x):
    return x[0]**2 + x[1]**2

# Restricción de igualdad: x1 + x2 = 1
restricciones = [
    {'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1},
    {'type': 'ineq', 'fun': lambda x: x[0] - x[1] - 1}
]

# Límites: x1, x2 ≥ 0
limites = [(0, None), (0, None)]

# Punto inicial
x0 = [0, 0]

# Resolver
resultado = minimize(objetivo, x0, constraints = restricciones, bounds=limites)

# Mostrar resultados
print("Punto óptimo:", resultado.x)
print("Valor mínimo:", resultado.fun)

for x0 in [[0, 0], [1, 1], [6, 2], [10, 10]]:
    res = minimize(objetivo, x0, constraints = restricciones)
    print(f"x0 = {x0} → x* = {res.x}, Z = {res.fun}")
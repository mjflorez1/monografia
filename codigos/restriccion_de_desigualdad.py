from scipy.optimize import minimize
import numpy as np

# Función a minimizar (negativo de la original)
def objetivo(x):
    return -2 * x[0] - 3 * x[1]

# Restricciones (forma: g(x) >= 0 → reescribimos)
restricciones = [
    {'type': 'ineq', 'fun': lambda x: 8 - x[0] - x[1]},     # x1 + x2 ≤ 8 → 8 - x1 - x2 ≥ 0
    {'type': 'ineq', 'fun': lambda x: 4 + x[0] - 2 * x[1]}, # -x1 + 2x2 ≤ 4 → 4 + x1 - 2x2 ≥ 0
    {'type': 'ineq', 'fun': lambda x: x[0]},
    {'type': 'ineq', 'fun': lambda x: x[1]}
]

# Límites: x1 ≥ 0, x2 ≥ 0
limites = [(0, None), (0, None)]

# Punto inicial
x0 = [0, 0]

# Resolver
resultado = minimize(objetivo, x0, method='SLSQP', bounds=limites, constraints=restricciones)

# Mostrar resultados
print("Punto óptimo:", resultado.x)
print("Valor mínimo:", resultado.fun)  # le cambiamos el signo para que sea el valor real

for x0 in [[0, 0], [1, 1], [6, 2], [10, 10]]:
    res = minimize(objetivo, x0, constraints=restricciones)
    print(f"x0 = {x0} → x* = {res.x}, Z = {-res.fun}")
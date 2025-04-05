import numpy as np
from scipy.optimize import minimize

#Definimos la función objetivo
def objective(x):
    x1, x2, x3 = x
    return x1**2 + x2**2 + x3**2 - 2*x1*x2

#Definimos las restricciones
def constraint1(x):
    return 2*x[0] + x[1] - 4  #2x1 + x2 - 4 = 0

def constraint2(x):
    return 5*x[0] - x[2] - 8  #5x1 - x3 - 8 = 0

#Agrupamos las restricciones en formato de diccionario
constraints = [{'type': 'eq', 'fun': constraint1},
               {'type': 'eq', 'fun': constraint2}]

#Valores iniciales para x1, x2, x3
x0 = np.array([0, 0, 0])  # Puedes cambiar estos valores

#Resolver el problema de optimización
solution = minimize(objective, x0, constraints=constraints)

#Imprimir resultados
print("Valores óptimos:", solution.x)
print("Valor mínimo de la función objetivo:", solution.fun)
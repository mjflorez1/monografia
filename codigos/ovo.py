import numpy as np
from scipy.optimize import minimize, linprog


# # Modelo cúbico y funciones f_i(x)
def model(t, x):
    return x[0] + x[1]*t + x[2]*t**2 + x[3]*t**3

def f_i(x, t_i, y_i):
    return 0.5 * (model(t_i, x) - y_i)**2

# def grad_f_i(x, t_i, y_i):
#     diff = model(t_i, x) - y_i
#     return diff * np.array([1, t_i, t_i**2, t_i**3])

# Construcción de I_delta(x)
def mount_Idelta(fovo,faux,indices,delta,Idelta):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1

    return k

    

# # Guardar restricciones
# def save_constraints(x, t, y, I_delta, iter_num, filename="restricciones.txt"):
#     with open(filename, "a") as f:
#         for j in I_delta:
#             grad = grad_f_i(x, t[j], y[j])
#             restr = " + ".join([f"{grad[k]:.4f}*x{k+1}" for k in range(len(grad))])
#             f.write(f"Iteración {iter_num} - Restricción {j}: {restr} <= w\n")
        
# # Subproblema usando linprog (simplificación)
# def solve_subproblem(xk, t, y, I_delta, delta=0.1):
#     n = len(xk)
#     A = []
#     b = []
#     for j in I_delta:
#         grad = grad_f_i(xk, t[j], y[j])
#         A.append(grad)
#         b.append(0.0)
#     A = np.array(A)
#     m = len(I_delta)

#     # LP: min w  s.t. g_j^T d <= w
#     c = np.zeros(n + 1)
#     c[-1] = 1  # minimize w

#     A_ub = np.hstack([A, -np.ones((m, 1))])
#     b_ub = np.zeros(m)

#     bounds = [(-delta, delta)] * n + [(None, None)]
#     res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

#     if res.success:
#         d = res.x[:-1]
#         w = res.x[-1]
#         return d, w
#     else:
#         raise RuntimeError("Subproblem did not converge")
# Algoritmo OVO simplificado

def ovo_algorithm(t,y):
    epsilon = 1e-3
    delta = 1e-3
    max_iter = 100
    max_iter_armijo = 100
    n = 5
    q = 40

    xk = np.array([-1,-2,1,-1])
    faux = np.zeros(m)
    Idelta = np.zeros(m,dtype=int)
    A = np.zeros(m,n)
    b = np.zeros(m)
    c = np.zeros(n)
    c[-1] = 1

    for i in range(m):
        faux[i] = f_i(xk,t[i],y[i])

    indices = np.argsort(faux)
    faux = np.sort(faux)
    fovo = faux[q]

    nconst = mount_Idelta(fovo,faux,indices,delta,Idelta)

    res = linprog(c, A_ub=A[0:nconst-1,:], b_ub=b[0:nconst-1], bounds=[x0_bounds,x1_bounds,x2_bounds,x3_bounds,x4_bounds])

    

data = np.loadtxt("data.txt")

t = data[:,0]
y = data[:,1]
m = len(t)
q = 36

ovo_algorithm(t,y)
# print("Solución encontrada:", xsol)
# print("Valor de la función OVO:", fval)
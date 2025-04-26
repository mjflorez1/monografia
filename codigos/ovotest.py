import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

def model(t, x):
    return x[0] + x[1]*t + x[2]*t**2 + x[3]*t**3

def f_i(x, t_i, y_i):
    return 0.5 * (model(t_i, x) - y_i)**2

def grad_f_i(x, t_i, y_i, grad):
    diff = model(t_i, x) - y_i
    grad[0] = 1.
    grad[1] = t_i
    grad[2] = t_i**2
    grad[3] = t_i**3
    return diff * grad[:]

def mount_Idelta(fovo, faux, indices, delta, Idelta):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

def ovo_algorithm(t, y):
    epsilon = 1e-3
    delta = 1e-3
    max_iter = 100
    max_iter_armijo = 20
    n = 5
    q = 1  # p=36 (q = p-1 para 0-based index)

    xk = np.array([-1.0, -2.0, 1.0, -1.0, 0.0])  # x4 dummy para LP
    m = len(t)
    
    for iter in range(max_iter):
        # Calcular f_i y ordenar
        faux = np.array([f_i(xk[:4], t[i], y[i]) for i in range(m)])
        indices = np.argsort(faux)
        sorted_faux = np.sort(faux)
        fovo = sorted_faux[q]
        
        # Montar I_epsilon
        Idelta = np.zeros(m, dtype=int)
        nconst = mount_Idelta(fovo, sorted_faux, indices, epsilon, Idelta)
        
        if nconst == 0:
            print("Sin restricciones activas.")
            break
        
        # Construir problema LP
        A = np.zeros((nconst, n))
        b = np.zeros(nconst)
        c = np.zeros(n)
        c[-1] = 1
        
        grad = np.zeros(4)
        for k in range(nconst):
            j = Idelta[k]
            diff_grad = grad_f_i(xk[:4], t[j], y[j], grad)
            A[k, :4] = diff_grad
            A[k, 4] = -1.0
        
        # Actualizar bounds dinámicamente
        deltax = 1.0
        bounds = [
            (max(-10.0 - xk[0], -deltax), min(10.0 - xk[0], deltax)),
            (max(-10.0 - xk[1], -deltax), min(10.0 - xk[1], deltax)),
            (max(-10.0 - xk[2], -deltax), min(10.0 - xk[2], deltax)),
            (max(-10.0 - xk[3], -deltax), min(10.0 - xk[3], deltax)),
            (None, 0.0)  # w <= 0
        ]
        
        # Resolver LP
        res = linprog(c, A_ub=A[:nconst], b_ub=b[:nconst], bounds=bounds, method='highs')
        
        if not res.success:
            print(f"Iter {iter}: Fallo en LP")
            break
        
        d = res.x[:4]
        w = res.x[4]
        
        # Criterio de parada
        if w >= -1e-8:
            print(f"Convergencia en iter {iter}")
            break
        
        # Búsqueda de línea de Armijo
        alpha = 0.1
        armijo_success = False
        for _ in range(max_iter_armijo):
            x_new = xk[:4] + alpha * d
            f_new = np.sort([f_i(x_new, t[i], y[i]) for i in range(m)])[q]
            
            if f_new <= fovo + 0.5 * alpha * w:
                armijo_success = True
                break
                
            alpha *= 0.5
            
        if not armijo_success:
            print(f"Iter {iter}: Fallo en Armijo")
            break
            
        xk[:4] = x_new
        print(f"Iter {iter}: f={f_new:.4f}, ||d||={np.linalg.norm(d):.4f}")
    
    # Graficar resultados
    t_vals = np.linspace(min(t), max(t), 100)
    y_model = model(t_vals, xk[:4])
    
    plt.scatter(t, y, c='blue', label='Datos')
    plt.plot(t_vals, y_model, 'k-', lw=2, label='Modelo OVO')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title(f'Ajuste OVO (q={q})')
    plt.show()
    
    return xk[:4]

# Cargar datos y ejecutar
data = np.loadtxt("data.txt")
t = data[:, 0]
y = data[:, 1]
m = len(t)

sol = ovo_algorithm(t, y)
print("Solución final:", np.round(sol, 4))
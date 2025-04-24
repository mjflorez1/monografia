import numpy as np
from scipy.optimize import linprog

def model(t, x):
    return x[0] + x[1]*t + x[2]*t**2 + x[3]*t**3

def f_i(x, t_i, y_i):
    return 0.5 * (model(t_i, x) - y_i)**2

def grad_f_i(x, t_i, y_i):
    diff = model(t_i, x) - y_i
    return diff * np.array([1, t_i, t_i**2, t_i**3])

def mount_Idelta(fovo, faux, indices, epsilon, Idelta):
    k = 0
    for i in range(len(faux)):
        if abs(fovo - faux[i]) <= epsilon:
            Idelta[k] = indices[i]
            k += 1
    return k

def ovo_algorithm(t, y):
    # Parámetros fijos (ajustables aquí)
    p = 36                  # p del paper (36 = m - 10 outliers)
    epsilon = 1e-3          # Tolerancia para I_epsilon
    delta = 1.0             # Tamaño región de confianza
    max_iter = 100          # Máximo de iteraciones
    max_iter_armijo = 20    # Máximo pasos Armijo
    theta = 0.5             # Parámetro Armijo
    sigma_min = 0.1         # Reducción mínima alpha
    sigma_max = 0.9         # Reducción máxima alpha
    
    # Inicialización
    q = p - 1               # Índice base 0
    m = len(t)
    xk = np.array([-1.0, -2.0, 1.0, -1.0])  # Punto inicial del paper
    print("Inicialización:", xk)
    
    for iter in range(max_iter):
        # Paso 1: Calcular f_ovo e I_epsilon
        faux = np.array([f_i(xk, t[i], y[i]) for i in range(m)])
        indices = np.argsort(faux)
        sorted_faux = np.sort(faux)
        fovo = sorted_faux[q]
        
        # Identificar restricciones activas
        Idelta = np.zeros(m, dtype=int)
        nconst = mount_Idelta(fovo, sorted_faux, indices, epsilon, Idelta)
        
        if nconst == 0:
            print("Sin restricciones activas.")
            break
        
        # Construir problema LP
        A = np.zeros((nconst, 5))  # 4 variables + w
        b = np.zeros(nconst)
        c = np.zeros(5)
        c[-1] = 1  # Minimizar w
        
        for k in range(nconst):
            j = Idelta[k]
            grad = grad_f_i(xk, t[j], y[j])
            A[k, :4] = grad
            A[k, 4] = -1.0  # Restricción: grad^T d <= w
        
        # Región de confianza (bounds)
        current_bounds = [
            (max(-10.0 - xk[0], min(10.0 - xk[0], delta))),
            (max(-10.0 - xk[1], min(10.0 - xk[1], delta))),
            (max(-10.0 - xk[2], min(10.0 - xk[2], delta))),
            (max(-10.0 - xk[3], min(10.0 - xk[3], delta))),
            (None, 0)  # w <= 0
        ]
        
        # Resolver subproblema LP
        res = linprog(c, A_ub=A[:nconst], b_ub=b[:nconst], bounds=current_bounds, method='highs')
        
        if not res.success:
            print(f"Iter {iter}: LP no converge")
            break
        
        d = res.x[:4]
        w = res.x[4]
        
        # Criterio de parada
        if w >= -1e-8:
            print(f"Iter {iter}: Convergencia (w={w:.3e})")
            break
        
        # Paso 2: Búsqueda de línea de Armijo
        alpha = 1.0
        armijo_iter = 0
        sufficient_decrease = False
        f_current = fovo
        
        while armijo_iter < max_iter_armijo:
            x_new = xk + alpha * d
            f_new = np.sort([f_i(x_new, t[i], y[i]) for i in range(m)])[q]
            
            if f_new <= f_current + theta * alpha * w:
                sufficient_decrease = True
                break
                
            # Reducir alpha
            alpha = alpha * np.random.uniform(sigma_min, sigma_max)
            armijo_iter += 1
        
        if not sufficient_decrease:
            print(f"Iter {iter}: Armijo falló")
            break
        
        # Actualizar iterado
        xk = x_new
        print(f"Iter {iter}: f={f_new:.4f}, ||d||={np.linalg.norm(d):.4f}, alpha={alpha:.4f}")
    
    return xk

# Cargar datos y ejecutar
data = np.loadtxt("data.txt")
t = data[:, 0]
y = data[:, 1]

solucion = ovo_algorithm(t, y)
print("\nSolución final:", solucion)
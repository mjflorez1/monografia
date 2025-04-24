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
    p = 36
    epsilon = 1e-3
    delta = 1.0
    max_iter = 100
    max_iter_armijo = 20
    theta = 0.5
    sigma_min = 0.1
    sigma_max = 0.9
    
    q = p - 1
    m = len(t)
    xk = np.array([-1.0, -2.0, 1.0, -1.0])
    print("Inicialización:", xk)
    
    for iter in range(max_iter):
        faux = np.array([f_i(xk, t[i], y[i]) for i in range(m)])
        indices = np.argsort(faux)
        sorted_faux = np.sort(faux)
        fovo = sorted_faux[q]
        
        Idelta = np.zeros(m, dtype=int)
        nconst = mount_Idelta(fovo, sorted_faux, indices, epsilon, Idelta)
        
        if nconst == 0:
            print("Sin restricciones activas.")
            break
        
        A = np.zeros((nconst, 5))
        b = np.zeros(nconst)
        c = np.zeros(5)
        c[-1] = 1
        
        for k in range(nconst):
            j = Idelta[k]
            grad = grad_f_i(xk, t[j], y[j])
            A[k, :4] = grad
            A[k, 4] = -1.0
        
        current_bounds = [
            (max(-delta, -10.0 - xk[0]), min(delta, 10.0 - xk[0])),
            (max(-delta, -10.0 - xk[1]), min(delta, 10.0 - xk[1])),
            (max(-delta, -10.0 - xk[2]), min(delta, 10.0 - xk[2])),
            (max(-delta, -10.0 - xk[3]), min(delta, 10.0 - xk[3])),
            (None, 0)
            ]
        
        res = linprog(c, A_ub=A[:nconst], b_ub=b[:nconst], bounds=current_bounds, method='highs')
        
        if not res.success:
            print(f"Iter {iter}: LP no converge")
            break
        
        d = res.x[:4]
        w = res.x[4]
        
        if w >= -1e-8:
            print(f"Iter {iter}: Convergencia (w={w:.3e})")
            break
        
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
                
            alpha = alpha * np.random.uniform(sigma_min, sigma_max)
            armijo_iter += 1
        
        if not sufficient_decrease:
            print(f"Iter {iter}: Armijo falló")
            break
        
        xk = x_new
        print(f"Iter {iter}: f={f_new:.4f}, ||d||={np.linalg.norm(d):.4f}, alpha={alpha:.4f}")
    
    return xk

data = np.loadtxt("data.txt")
t = data[:, 0]
y = data[:, 1]

solucion = ovo_algorithm(t, y)
print("\nSolución final:", solucion)
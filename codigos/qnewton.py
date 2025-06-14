# Bibliotecas importantes
import numpy as np

# Definicion del modelo cubico
def model(t,x1,x2,x3,x4):
    res = x1 + (x2 * t) + x3 * (t**2) + x4 * (t**3)
    return res

# Funciones de error cuadratico
def f_i(t_i,y_i,x):
    res = 0.5 * ((model(t_i,*x) - y_i)**2)
    return res

# Gradientes de las funciones de error
def grad_f_i(t_i,y_i,x,grad):
    diff = model(t_i,*x) - y_i
    grad[0] = diff * 1
    grad[1] = diff * t_i
    grad[2] = diff * t_i**2
    grad[3] = diff * t_i**3
    return grad[:]

# Hessiana de las funciones de error
def hess_f_i(t_i, y_i, x, H):
    a = [1, t_i, t_i**2, t_i**3]
    for i in range(4):
        for j in range(i, 4):
            H[i, j] = a[i] * a[j]
            H[j, i] = H[i, j]
    return H[:]

# Montamos el conjundo de indices I_delta
def mount_Idelta(fovo,faux,indices,delta,Idelta,m):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

# Computamos Bkj con ajuste de autovalores
def compute_Bkj(H, epsilon):
    eigvals = np.linalg.eigvalsh(H)
    lambda_min = np.min(eigvals)
    ajuste = max(0, -lambda_min + epsilon)
    Bkj = H + ajuste * np.eye(H.shape[0])
    return Bkj

# Resolver el subproblema cuadrático de manera más simple y efectiva
def solve_quadratic_subproblem(G, Bk, deltax):
    """
    Resuelve el subproblema usando la estrategia del gradiente promedio
    con información de segunda derivada (Hessiana)
    """
    nconst, n = G.shape
    
    # Calcular gradiente promedio ponderado
    grad_avg = np.mean(G, axis=0)
    
    # Dirección de Newton con el gradiente promedio
    try:
        d = -np.linalg.solve(Bk, grad_avg)
    except np.linalg.LinAlgError:
        # Si la matriz no es invertible, usar descenso de gradiente
        d = -grad_avg
    
    # Proyectar a las restricciones de caja
    d_proj = np.zeros_like(d)
    for j in range(len(d)):
        d_proj[j] = np.clip(d[j], -deltax, deltax)
    
    # Calcular el valor del modelo (debe ser negativo para descenso)
    mkd = np.dot(grad_avg, d_proj) + 0.5 * np.dot(d_proj, Bk @ d_proj)
    
    return d_proj, mkd

def ovo_newton(t,y):
    # Parametros algoritmicos (iguales al código con linprog)
    epsilon = 1e-8
    delta   = 1e-3
    deltax  = 1.0
    theta   = 0.5
    n = 5
    q = 35
    max_iter = 1000
    max_iter_armijo = 100
    iter = 1
    
    # Solucion inicial (igual al código con linprog)
    xk = np.array([-1,-2,1,-1])
    
    # Definimos algunos arrays necesarios
    xktrial = np.zeros(n-1)
    faux    = np.zeros(m)
    Idelta  = np.zeros(m,dtype=int)
    grad    = np.zeros(n-1)
    Htemp   = np.zeros((n-1, n-1))
    
    while iter <= max_iter:
        
        iter_armijo = 0
        
        # Evaluar funciones (igual al código con linprog)
        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)

        # Ordenar por valores crecientes (igual al código con linprog)
        indices = np.argsort(faux)
        faux = np.sort(faux)
        fxk = faux[q]

        # Construir I_delta (igual al código con linprog)
        nconst = mount_Idelta(fxk, faux, indices, delta, Idelta, m)
        
        if nconst == 0:
            print("No hay restricciones activas")
            break

        # Construir matriz de gradientes (igual al código con linprog)
        G = np.zeros((nconst, n-1))
        for i in range(nconst):
            ind = Idelta[i]
            grad_f_i(t[ind], y[ind], xk, grad)
            G[i, :] = grad[:]

        # Hessiana promedio del conjunto activo
        H_prom = np.zeros((n-1, n-1))
        for i in range(nconst):
            idx = Idelta[i]
            hess_f_i(t[idx], y[idx], xk, Htemp)
            H_prom += Htemp
        H_prom /= nconst

        # Calcular Bk con ajuste de autovalores
        Bk = compute_Bkj(H_prom, epsilon)

        # Resolver subproblema cuadrático (equivalente al linprog)
        dk, mkd = solve_quadratic_subproblem(G, Bk, deltax)
        
        # Si no hay descenso, probar con gradiente de máxima norma
        if mkd >= 0:
            # Encontrar el gradiente con mayor norma
            max_norm = 0
            best_grad_idx = 0
            for i in range(nconst):
                norm_g = np.linalg.norm(G[i, :])
                if norm_g > max_norm:
                    max_norm = norm_g
                    best_grad_idx = i
            
            # Usar solo ese gradiente
            g_best = G[best_grad_idx, :]
            try:
                dk = -np.linalg.solve(Bk, g_best)
            except np.linalg.LinAlgError:
                dk = -g_best / np.linalg.norm(g_best)
            
            # Proyectar a restricciones de caja
            for j in range(len(dk)):
                dk[j] = np.clip(dk[j], -deltax, deltax)
            
            mkd = np.dot(g_best, dk) + 0.5 * np.dot(dk, Bk @ dk)
        
        print(fxk, mkd, iter, iter_armijo)
        
        # Criterio de parada (igual al código con linprog)
        if abs(mkd) < epsilon:
            break
        
        # Búsqueda lineal tipo Armijo (igual al código con linprog)
        alpha = 1.0

        while iter_armijo <= max_iter_armijo:
            
            iter_armijo += 1
            
            xktrial = xk + alpha * dk
            
            # Calcular nuevo valor OVO (igual al código con linprog)
            for i in range(m):
                faux[i] = f_i(t[i], y[i], xktrial)
            faux = np.sort(faux)
            fxktrial = faux[q]
            
            # Condición de Armijo (igual al código con linprog)
            if fxktrial < fxk + theta * alpha * mkd:
                break
            
            alpha = 0.5 * alpha

        # Actualizar iterado (igual al código con linprog)
        xk = xktrial
        iter += 1
        
    print(xk)

data = np.loadtxt("data.txt")
t = data[:,0]
y = data[:,1]
m = len(t)

ovo_newton(t,y)
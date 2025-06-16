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

def ovo_newton(t,y):
    # Parametros algoritmicos
    epsilon = 1e-8
    delta   = 1e-3
    deltax  = 1.0
    theta   = 0.5
    n = 5
    q = 35
    max_iter = 1000
    max_iter_armijo = 100
    iter = 1
    
    # Solucion inicial
    xk = np.array([-1,-2,1,-1])
    
    # Definimos algunos arrays necesarios
    xktrial = np.zeros(n-1)
    faux    = np.zeros(m)
    Idelta  = np.zeros(m,dtype=int)
    grad    = np.zeros(n-1)
    Htemp   = np.zeros((n-1, n-1))
    
    while iter <= max_iter:
        # Evaluar funciones
        for i in range(m):
            faux[i] = f_i(t[i], y[i], xk)

        # Ordenar por valores crecientes
        indices = np.argsort(faux)
        faux = np.sort(faux)
        fxk = faux[q]

        # Construir I_delta
        nconst = mount_Idelta(fxk, faux, indices, delta, Idelta,m)

        # Matrices de gradientes y Hessianas
        A = np.zeros((nconst, n))
        b = np.zeros(nconst)
        G = np.zeros((nconst, n-1))
        
        for i in range(nconst):
            idx = Idelta[i]
            grad_f_i(t[idx], y[idx], xk, grad)
            A[i, :-1] = grad[:]
            A[i, -1] = -1
            G[i, :] = grad[:]

        # Hessiana promedio
        H_prom = np.zeros((n-1, n-1))
        for i in range(nconst):
            idx = Idelta[i]
            hess_f_i(t[idx], y[idx], xk, Htemp)
            H_prom += Htemp
        H_prom /= nconst

        # Calcular Bk con autovalores
        Bk = compute_Bkj(H_prom, epsilon)

        # Selección de gradiente "peor"
        Mkdk = np.inf
        for i in range(nconst):
            g = G[i, :]
            dtemp = -np.linalg.solve(Bk, g)
            mk_i = np.dot(g, dtemp) + 0.5 * np.dot(dtemp, Bk @ dtemp)
            if mk_i < Mkdk:  # Buscamos la MAYOR reducción
                Mkdk = mk_i
                gsel = g.copy()
                jsel = Idelta[i]

        # Dirección de descenso
        dk = -np.linalg.solve(Bk, gsel)
        
        # Búsqueda tipo Armijo
        alpha = 1.0
        fxo = np.sort([f_i(t[i], y[i], xk) for i in range(m)])[q]

        for j in range(max_iter_armijo):
            xtrial = xk + alpha * dk
            # Calcular NUEVO valor OVO
            fxtrial = np.sort([f_i(t[i], y[i], xtrial) for i in range(m)])[q]
            
            # Condición de Armijo para el valor OVO
            if fxtrial <= fxo + theta * alpha * Mkdk:
                break
            alpha *= 0.5

        # Criterio de parada
        print(fxk, Mkdk, iter, max_iter_armijo)
        if np.linalg.norm(dk) < epsilon or np.abs(fxk - fxtrial) < epsilon:
            break

        # Actualizacion iterado
        xk = xk + alpha * dk
        iter += 1
        
    print(xk)

data = np.loadtxt("data.txt")
t = data[:,0]
y = data[:,1]
m = len(t)

ovo_newton(t,y)
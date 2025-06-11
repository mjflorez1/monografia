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
def hess_f_i(t_i,y_i,x,H):
    a0 = 1
    a1 = t_i
    a2 = t_i**2
    a3 = t_i**3

    H[0,0] = a0 * a0
    H[0,1] = a0 * a1
    H[0,2] = a0 * a2
    H[0,3] = a0 * a3

    H[1,0] = H[0,1]
    H[1,1] = a1 * a1
    H[1,2] = a1 * a2
    H[1,3] = a1 * a3

    H[2,0] = H[0,2]
    H[2,1] = H[1,2]
    H[2,2] = a2 * a2
    H[2,3] = a2 * a3

    H[3,0] = H[0,3]
    H[3,1] = H[1,3]
    H[3,2] = H[2,3]
    H[3,3] = a3 * a3
    return H[:]

hess_f_i(t_i,y_i,x,H)
autovalores = np.linalg.eig(H[:])

# Montamos el conjundo de indices I_delta
def mount_Idelta(fovo,faux,indices,delta,Idelta):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

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

data = np.loadtxt("data.txt")
t = data[:,0]
y = data[:,1]
m = len(t)

ovo_newton(t,y)
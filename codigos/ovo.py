import numpy as np
from scipy.optimize import minimize, linprog

def model(t, x):
    return x[0] + x[1]*t + x[2]*t**2 + x[3]*t**3

def f_i(x, t_i, y_i):
    return 0.5 * (model(t_i, x) - y_i)**2

def grad_f_i(x, t_i, y_i):
    diff = model(t_i, x) - y_i
    return diff * np.array([1, t_i, t_i**2, t_i**3])

def mount_Idelta(fovo,faux,indices,delta,Idelta):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

def ovo_algorithm(t,y):
    epsilon = 1e-3
    delta = 1e-3
    max_iter = 100
    max_iter_armijo = 100
    n = 5
    q = 40

    xk = np.array([-1,-2,1,-1, 0])
    faux = np.zeros(m)
    Idelta = np.zeros(m,dtype=int)
    A = np.zeros((m,n))
    b = np.zeros(m)
    c = np.zeros(n)
    c[-1] = 1
    deltax = 1.0
    
    x0_bounds = [max(-10 - xk[0], -deltax), min(10 - xk[0], deltax)]
    x1_bounds = [max(-10 - xk[1], -deltax), min(10 - xk[1], deltax)]
    x2_bounds = [max(-10 - xk[2], -deltax), min(10 - xk[2], deltax)]
    x3_bounds = [max(-10 - xk[3], -deltax), min(10 - xk[3], deltax)]
    x4_bounds = [None, 0]

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
#q = 35

ovo_algorithm(t,y)
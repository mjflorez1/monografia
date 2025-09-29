import numpy as np
from scipy.optimize import linprog
from tabulate import tabulate

def model(t_i,x0,x1,x2):
    return x0 * np.exp(x1 / (t_i + x2))

def f_i(t_i, y_i, x):
    return 0.5 * ((model(t_i,*x) - y_i) ** 2)

def grad_f_i(t_i,y_i,x,grad):
    x0, x1, x2 = x
    z = np.exp(x1 / (t_i + x2))
    diff = (x0 * z) - y_i
    grad[0] = z
    grad[1] = x0 * z / (t_i + x2)
    grad[2] = -x0 * x1 * z / ((t_i + x2)**2)
    return diff * grad[:]

def mount_Idelta(fovo, faux, indices, epsilon, Idelta, m):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= epsilon:
            Idelta[k] = indices[i]
            k += 1
    return k

def ovo(t, y):
    stop = 1e-8
    epsilon = 1e+2
    delta = 1e+12
    theta = 1e-12
    n = 4
    q = 15
    max_iter = 1
    max_iter_armijo = 1000
    iter = 1
    m = len(t)
    alpha_k = 1

    xk = np.array([0.02, 4000, 250])
    xktrial = np.zeros(n - 1)
    faux = np.zeros(m)
    Idelta = np.empty(m, dtype=int)

    header = ["f(xk)","Iter1","Iter2", "Mk(d)","ncons","Idelta"]
    table = []

    # La función objetivo es lineal y depende unicamente 
    # de la variable artificial
    c = np.zeros(n)
    c[-1] = 1

    while iter <= max_iter:
        iter_armijo = 0

        # Restricciones de caja
        x0_bounds = (-delta,delta)
        x1_bounds = (-delta,delta)
        x2_bounds = (-delta,delta)
        x3_bounds = (None,0)

        # Calculamos de las funciones de error
        for i in range(m):
            faux[i] = f_i(t[i],y[i],xk)

        print(faux)

        exit

        # Ordenamos las funciones de error y sus respectivos indices
        indices = np.argsort(faux)
        faux = np.sort(faux)

        # Funcion de valor ordenado de orden q
        fxk = faux[q]

        # Computamos Idelta para saber el numero de restricciones
        nconst = mount_Idelta(fxk,faux,indices,epsilon,Idelta,m)

        # Montamos la matriz de restricciones 
        A = np.zeros((nconst,n))
        b = np.zeros(nconst)
        grad = np.zeros((nconst,n-1))

        for i in range(nconst):
            ind = Idelta[i]
            grad_f_i(t[ind],y[ind],xk,grad[i,:])

            A[i,:-1] = grad[i,:]
            A[i,-1] = -1

        res = linprog(c,A_ub=A,b_ub=b,bounds=[x0_bounds,x1_bounds,x2_bounds,x3_bounds])

        # Solucion del subproblema convexo
        dk = res.x
        mkd = dk[-1]

        if abs(mkd) < stop:
            break

        while iter_armijo <= max_iter_armijo:
            
            iter_armijo += 1
            
            xktrial = xk + (alpha_k * dk[:-1])
            
            for i in range(m):
                faux[i] = f_i(t[i], y[i], xktrial)
                
            faux = np.sort(faux)
            fxktrial = faux[q]
            
            if fxktrial < fxk + (theta * alpha_k * mkd):
                break
            
            alpha_k *= 0.5

        print(fxktrial,fxk + (theta * alpha_k * mkd))

        table.append([fxk,iter,iter_armijo,mkd,nconst,Idelta[:nconst]])
        
        xk = xktrial
        iter += 1
        alpha_k = 1


    print(tabulate(table, headers=header, tablefmt="grid"))
    print('Solución final:', xk)





data = np.loadtxt('meyer_data.txt')
t = data[:, 0]
y = data[:, 1]
ovo(t, y)
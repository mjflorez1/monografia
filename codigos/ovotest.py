import numpy as np
from scipy.optimize import minimize, linprog

def model(t, x):
    return x[0] + x[1]*t + x[2]*t**2 + x[3]*t**3

def f_i(x, t_i, y_i):
    return 0.5 * (model(t_i, x) - y_i)**2

def grad_f_i(x,t_i,y_i,grad):
    diff = model(t_i, x) - y_i
    
    grad[0] = 1.
    grad[1] = t_i
    grad[2] = t_i**2
    grad[3] = t_i**3

    return grad

def mount_Idelta(fovo,faux,indices,delta,Idelta):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

def ovo_algorithm(t,y):
    epsilon = 1e-3
    delta = 1e-2
    max_iter = 100
    max_iter_armijo = 1000
    n = 5
    q = 40

    xk = np.array([-1,-2,1,-1, 0])
    faux = np.zeros(m)
    Idelta = np.zeros(m,dtype=int)
    grad = np.zeros((m,n-1))
    A = np.zeros((m,n))
    b = np.zeros(m)
    c = np.zeros(n)
    c[-1] = 1
    deltax = 1.0
    theta = 0.5

    while True:    
    
        x0_bounds = [max(-10 - xk[0], -deltax), min(10 - xk[0], deltax)]
        x1_bounds = [max(-10 - xk[1], -deltax), min(10 - xk[1], deltax)]
        x2_bounds = [max(-10 - xk[2], -deltax), min(10 - xk[2], deltax)]
        x3_bounds = [max(-10 - xk[3], -deltax), min(10 - xk[3], deltax)]
        x4_bounds = [None, 0]

        for i in range(m):
            faux[i] = f_i(xk,t[i],y[i])

        indices = np.argsort(faux)
        faux = np.sort(faux)
        fxk = faux[q]

        nconst = mount_Idelta(fxk,faux,indices,delta,Idelta)

        for i in range(nconst):
            ind = Idelta[i]
            grad_f_i(xk,t[ind],y[ind],grad[i,:])

            A[i,0:n-1] = grad[i,:]
            A[i,n-1] = -1

        res = linprog(c, A_ub=A[0:nconst,:],b_ub=b[0:nconst],bounds=[x0_bounds,x1_bounds,x2_bounds,x3_bounds,x4_bounds])

        dk = res.x
        stop_criteria = 0

        # --> Criterio de parada <--

        for i in range(nconst):
            mkd = np.dot(grad[i,:],dk[0:n-1])
            print(mkd)

            if mkd >= stop_criteria: 
                stop_criteria = mkd

        if abs(mkd) < epsilon:
            break

        # --> Condición de descenso (Armijo) <--

        alpha = 1
        xktrial = xk[0:n-1] + alpha * dk[0:n-1]

        for i in range(m):
            faux[i] = f_i(xktrial,t[i],y[i])

        indices = np.argsort(faux)
        faux = np.sort(faux)
        fxktrial = faux[q]

        iter_armijo = 0

        while fxktrial >= fxk and iter_armijo <= max_iter_armijo:
            print("Iter Armijo:", iter_armijo)
            print("alpha:", alpha)
            print("fxk:", fxk)
            print("fxktrial:", fxktrial)
            print("theta * alpha * mkd:", theta * alpha * mkd)
            print("Armijo RHS:", fxk + theta * alpha * mkd)
            print("---")
            alpha = 0.5 * alpha

            xktrial = xk[0:n-1] + alpha * dk[0:n-1]

            for i in range(m):
                faux[i] = f_i(xktrial,t[i],y[i])

            indices = np.argsort(faux)
            faux = np.sort(faux)
            fxktrial = faux[q]
            iter_armijo += 1

        xk = xk[0:n-1] + alpha * dk[0:n-1]
        xk[-1] = 0

        print(fxk,fxktrial,iter_armijo)


        break




        


data = np.loadtxt("data.txt")

t = data[:,0]
y = data[:,1]
m = len(t)
#q = 35

ovo_algorithm(t,y)
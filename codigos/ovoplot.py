# Bibliotecas esenciales
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

size_img = 0.6
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = [size_img * 6.4,size_img * 4.8]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Definición del modelo cubico
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

# Montamos el conjundo de indices I_delta
def mount_Idelta(fovo,faux,indices,delta,Idelta):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

def ovo_algorithm(t,y):

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

    # La función objetivo es lineal y depende unicamente 
    # de la variable artificial
    c = np.zeros(n)
    c[-1] = 1


    while iter <= max_iter:    

        iter_armijo = 0

        # Restricciones de caja
        x0_bounds = (max(-10 - xk[0], -deltax), min(10 - xk[0], deltax))
        x1_bounds = (max(-10 - xk[1], -deltax), min(10 - xk[1], deltax))
        x2_bounds = (max(-10 - xk[2], -deltax), min(10 - xk[2], deltax))
        x3_bounds = (max(-10 - xk[3], -deltax), min(10 - xk[3], deltax))
        x4_bounds = (None, 0)

        # Calculamos de las funciones de error
        for i in range(m):
            faux[i] = f_i(t[i],y[i],xk)

        # Ordenamos las funciones de error y sus respectivos indices
        indices = np.argsort(faux)
        faux = np.sort(faux)

        # Funcion de valor ordenado de orden q
        fxk = faux[q]

        # Computamos Idelta para saber el numero de restricciones
        nconst = mount_Idelta(fxk,faux,indices,delta,Idelta)

        # Montamos la matriz de restricciones 
        A = np.zeros((nconst,n))
        b = np.zeros(nconst)
        grad = np.zeros((nconst,n-1))

        for i in range(nconst):
            ind = Idelta[i]
            grad_f_i(t[ind],y[ind],xk,grad[i,:])

            A[i,:-1] = grad[i,:]
            A[i,-1] = -1

        res = linprog(c,A_ub=A,b_ub=b,bounds=[x0_bounds,x1_bounds,x2_bounds,x3_bounds,x4_bounds])

        # Solucion del subproblema convexo
        dk = res.x
        mkd = dk[-1]

        #print(fxk,mkd,iter,iter_armijo)

        # Criterio de parada        
        if abs(mkd) < epsilon:
            break

        alpha = 1

        while iter_armijo <= max_iter_armijo:

            iter_armijo += 1
            
            xktrial = xk + (alpha * dk[:-1])

            for i in range(m):
                faux[i] = f_i(t[i],y[i],xktrial)

            faux = np.sort(faux)
            fxktrial = faux[q]

            if fxktrial < fxk + (theta * alpha * mkd):
                break

            alpha = 0.5 * alpha
            
        print(iter,fxk,mkd,iter_armijo)

        xk = xktrial
        iter += 1
        np.savetxt('txt/sol_cubic_cauchy.txt',xk, fmt="%.6f")

    print('Solución final: ',xk)
    return xk, fxk    

data = np.loadtxt("txt/data.txt")

t = data[:,0]
y = data[:,1]
m = len(t)

# Arrays para almacenar resultados
outliers_list = []
fxk_list = []

# Ejecutar para diferentes números de outliers
for num_outliers in range(0, 16):
    print(f"\n{'='*60}")
    print(f"Ejecutando con {num_outliers} outliers")
    print(f"{'='*60}")
    
    # Modificar q según el número de outliers
    q_original = 35
    q = m - num_outliers - 1
    
    # Crear copia del algoritmo con q modificado
    epsilon = 1e-8
    delta   = 1e-3
    deltax  = 1.0
    theta   = 0.5
    n = 5
    max_iter = 1000
    max_iter_armijo = 100
    iter = 1

    xk = np.array([-1,-2,1,-1])
    xktrial = np.zeros(n-1)
    faux    = np.zeros(m)
    Idelta  = np.zeros(m,dtype=int)
    c = np.zeros(n)
    c[-1] = 1

    while iter <= max_iter:    
        iter_armijo = 0

        x0_bounds = (max(-10 - xk[0], -deltax), min(10 - xk[0], deltax))
        x1_bounds = (max(-10 - xk[1], -deltax), min(10 - xk[1], deltax))
        x2_bounds = (max(-10 - xk[2], -deltax), min(10 - xk[2], deltax))
        x3_bounds = (max(-10 - xk[3], -deltax), min(10 - xk[3], deltax))
        x4_bounds = (None, 0)

        for i in range(m):
            faux[i] = f_i(t[i],y[i],xk)

        indices = np.argsort(faux)
        faux = np.sort(faux)
        fxk = faux[q]

        nconst = mount_Idelta(fxk,faux,indices,delta,Idelta)

        A = np.zeros((nconst,n))
        b = np.zeros(nconst)
        grad = np.zeros((nconst,n-1))

        for i in range(nconst):
            ind = Idelta[i]
            grad_f_i(t[ind],y[ind],xk,grad[i,:])
            A[i,:-1] = grad[i,:]
            A[i,-1] = -1

        res = linprog(c,A_ub=A,b_ub=b,bounds=[x0_bounds,x1_bounds,x2_bounds,x3_bounds,x4_bounds])

        dk = res.x
        mkd = dk[-1]

        if abs(mkd) < epsilon:
            break

        alpha = 1

        while iter_armijo <= max_iter_armijo:
            iter_armijo += 1
            xktrial = xk + (alpha * dk[:-1])

            for i in range(m):
                faux[i] = f_i(t[i],y[i],xktrial)

            faux = np.sort(faux)
            fxktrial = faux[q]

            if fxktrial < fxk + (theta * alpha * mkd):
                break

            alpha = 0.5 * alpha
            
        xk = xktrial
        iter += 1
    
    outliers_list.append(num_outliers)
    fxk_list.append(fxk)
    print(f'Solución con {num_outliers} outliers, f(x*) = {fxk}')

plt.plot(outliers_list, fxk_list, 'o-', linewidth=1, markersize=3)
plt.xlabel('Número de outliers ($o$)', fontsize=12)
plt.ylabel('$f(x^*)$', fontsize=12)
plt.yscale('log')
plt.savefig("figuras/cubic_cauchy_outs.pdf", bbox_inches="tight")
plt.show()
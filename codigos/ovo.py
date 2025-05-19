"""
Algoritmo de Optimización por Valor de Orden (OVO) – Método tipo Cauchy
------------------------------------------------------------------------
Esta implementación está basada en el artículo:

  Roberto Andreani, Cibele Dunder, José Mario Martínez.
  "Order-Value Optimization: Formulation and solution by means of a primal cauchy method".
  IMECC-UNICAMP e IME-USP, Brasil, 2003.

Resumen:
  Este código implementa un método iterativo para resolver el problema de Optimización por Valor de Orden (OVO),
  una generalización del problema clásico de Minimax. El objetivo es minimizar el valor funcional que ocupa la 
  posición p-ésima dentro de un conjunto dado de funciones. Se utiliza un método tipo Cauchy que garantiza 
  convergencia a puntos que satisfacen condiciones de optimalidad adecuadas, incluso en presencia de outliers.

Implementación en Python realizada por:
  Mateo Flores  
  Estudiante del programa de Matemáticas  
  Universidad del Atlántico

Orientador:
  Dr. Gustavo Quintero  
  Tutor de la monografía de grado  
  Universidad del Atlántico
"""

# Bibliotecas esenciales
import numpy as np
from scipy.optimize import linprog

# Definición del modelo cubico
def model(t,x1,x2,x3,x4):
    return x1 + (x2 * t) + (x3 * t)**2 + (x4 * t)**3

# Funciones de error cuadratico
def f_i(t_i,y_i,x):
    return 0.5 * (model(t_i,*x) - y_i)**2

# Gradientes de las funciones de error
def grad_f_i(t_i,y_i,x,grad):
    diff = model(t_i,*x) - y_i
    
    grad[0] = 1.
    grad[1] = t_i
    grad[2] = t_i**2
    grad[3] = t_i**3

    return diff * grad[:]


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
    epsilon = 1e-3
    delta = 1e-2
    max_iter = 10
    max_iter_armijo = 100
    n = 5
    q = 36
    deltax = 1.0
    theta = 0.5
    iter = 0

    # Solucion inicial
    xk = np.array([-1,-2,1,-1])

    # Definicion de algunos arrays necesarios
    faux = np.zeros(m)
    Idelta = np.zeros(m,dtype=int)
    grad = np.zeros((m,n-1))
    c = np.zeros(n)
    xktrial = np.zeros(n-1)

    # La función objetivo es lineal y depende unicamente 
    # de la variable artificial
    c[-1] = 1

    while True:    

        iter += 1
    
        x0_bounds = (max(-10 - xk[0], -deltax), min(10 - xk[0], deltax))
        x1_bounds = (max(-10 - xk[1], -deltax), min(10 - xk[1], deltax))
        x2_bounds = (max(-10 - xk[2], -deltax), min(10 - xk[2], deltax))
        x3_bounds = (max(-10 - xk[3], -deltax), min(10 - xk[3], deltax))
        x4_bounds = (None, 0)

        for i in range(m):
            faux[i] = f_i(t[i],y[i],xk)

        indices = np.argsort(faux)
        faux = np.sort(faux)

        # Funcion de valor ordenado
        fxk = faux[q-1]

        nconst = mount_Idelta(fxk,faux,indices,delta,Idelta)

        A = np.zeros((nconst,n))
        b = np.zeros(nconst)

        for i in range(nconst):
            ind = indices[i]
            grad_f_i(t[ind],y[ind],xk,grad[i,:])

            A[i,:-1] = grad[i,:]
            A[i,-1] = -1

        res = linprog(c,A_ub=A,b_ub=b,bounds=[x0_bounds,x1_bounds,x2_bounds,x3_bounds,x4_bounds])

        dk = res.x
        mkd = dk[-1]

        # Criterio de parada        
        if abs(mkd) < epsilon or iter >= max_iter:
            break

        iter_armijo = 0
        alpha = 1

        while True:
            
            xktrial = xk + alpha * dk[:-1]

            for i in range(m):
                faux[i] = f_i(t[i],y[i],xktrial)

            faux = np.sort(faux)
            fxktrial = faux[q-1]
            iter_armijo += 1

            if fxktrial >= fxk + theta * alpha * mkd and iter_armijo < max_iter_armijo:
                break

            alpha = 0.5 * alpha

        xk = xktrial + alpha * dk[:-1]

        # print(fxk,iter,iter_armijo)



data = np.loadtxt("data.txt")

t = data[:,0]
y = data[:,1]
m = len(t)

ovo_algorithm(t,y)
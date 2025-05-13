import numpy as np
from scipy.optimize import minimize, linprog

def model(t, x): 
    return x[0] + x[1]*t + x[2]*t**2 + x[3]*t**3

def f_i(x, t_i, y_i): 
    return 0.5 * (model(t_i, x) - y_i)**2

def grad_f_i(x, t_i, y_i, grad):
    diff = model(t_i, x) - y_i
    
    grad[0] = 1.0
    grad[1] = t_i
    grad[2] = t_i**2
    grad[3] = t_i**3
    
    return diff * grad[:]

def mount_Idelta(fovo, faux, indices, delta, Idelta):
    k = 0
    for i in range(m):
        if abs(fovo - faux[i]) <= delta:
            Idelta[k] = indices[i]
            k += 1
    return k

def ovo_algorithm(t, y):
    epsilon = 1e-3
    delta = 1e-3
    max_iter = 100
    max_iter_armijo = 1000
    n = 5  # 4 parámetros del modelo + 1 variable slack
    q = 40
    
    xk = np.array([-1, -2, 1, -1, 0])
    faux = np.zeros(m)
    Idelta = np.zeros(m, dtype=int)
    grad = np.zeros((m, n-1))
    A = np.zeros((m, n))
    b = np.zeros(m)
    c = np.zeros(n)
    c[-1] = 1  # Minimizar la variable slack
    deltax = 1.0
    theta = 0.5
    iter_count = 0

    while iter_count < max_iter:    
        iter_count += 1
        
        # Límites para las variables
        x0_bounds = [max(-10 - xk[0], -deltax), min(10 - xk[0], deltax)]
        x1_bounds = [max(-10 - xk[1], -deltax), min(10 - xk[1], deltax)]
        x2_bounds = [max(-10 - xk[2], -deltax), min(10 - xk[2], deltax)]
        x3_bounds = [max(-10 - xk[3], -deltax), min(10 - xk[3], deltax)]
        x4_bounds = [None, 0]  # La variable slack es no positiva

        # Calcula los valores de la función en el punto actual
        for i in range(m):
            faux[i] = f_i(xk[:n-1], t[i], y[i])

        # Ordena los valores de la función
        indices = np.argsort(faux)
        faux = np.sort(faux)
        fxk = faux[q-1] if q <= m else faux[-1]  # Obtener el q-ésimo valor

        # Encuentra los índices dentro de delta del q-ésimo valor
        nconst = mount_Idelta(fxk, faux, indices, delta, Idelta)

        # Configura las restricciones para el subproblema
        for i in range(nconst):
            ind = Idelta[i]
            grad_f_i(xk[:n-1], t[ind], y[ind], grad[i, :])
            
            # La matriz A para el subproblema LP
            A[i, 0:n-1] = grad[i, :]
            A[i, n-1] = -1
            
            # El lado derecho de las restricciones (b) es cero
            b[i] = 0

        # Resuelve el subproblema de programación lineal
        res = linprog(c, A_ub=A[0:nconst, :], b_ub=b[0:nconst], 
                      bounds=[x0_bounds, x1_bounds, x2_bounds, x3_bounds, x4_bounds],
                      method='highs')  # Usa el solver 'highs' que es más robusto

        # Extrae la dirección y el valor objetivo del subproblema
        dk = res.x
        mk_dk = res.x[-1]  # Este es el valor de w, que debería ser negativo para una dirección de descenso
        
        # Criterio de parada
        if abs(mk_dk) < epsilon:
            print(f"Convergencia alcanzada: |mk_dk| = {abs(mk_dk):.6e} < {epsilon}")
            break

        # Búsqueda de línea con la condición de Armijo
        alpha = 1.0
        iter_armijo = 0
        success = False

        while iter_armijo < max_iter_armijo and not success:
            # Calcula el punto de prueba
            xktrial = xk[:n-1].copy()  # Copia para evitar modificar xk directamente
            xktrial = xktrial + alpha * dk[:n-1]
            
            # Evalúa la función en el punto de prueba
            for i in range(m):
                faux[i] = f_i(xktrial, t[i], y[i])
            
            # Ordena los valores de la función
            faux_sorted = np.sort(faux)
            fxktrial = faux_sorted[q-1] if q <= m else faux_sorted[-1]
            
            # Verifica la condición de Armijo: f(x_k + α*d_k) ≤ f(x_k) + θ*α*M_k(d_k)
            if fxktrial <= fxk + theta * alpha * mk_dk:
                success = True
            else:
                # Reduce el tamaño del paso
                alpha *= 0.5
                iter_armijo += 1
        
        if not success:
            print(f"Advertencia: La búsqueda de línea de Armijo alcanzó el máximo de iteraciones ({max_iter_armijo})")
            if alpha < 1e-10:
                print("El tamaño del paso se volvió demasiado pequeño. Terminando.")
                break
        
        # Actualiza el punto actual
        xk[:n-1] = xk[:n-1] + alpha * dk[:n-1]
        
        print(f"Iter {iter_count}: f(xk)={fxk:.6e}, f(xk+1)={fxktrial:.6e}, alpha={alpha:.6e}, iter_armijo={iter_armijo}")
        
        # Criterio de parada adicional si la mejora es muy pequeña
        if abs(fxk - fxktrial) < epsilon * abs(fxk):
            print(f"Convergencia: Mejora relativa en la función < {epsilon}")
            break

    return xk[:n-1]

# Código para cargar datos y ejecutar el algoritmo
data = np.loadtxt("data.txt")
t = data[:,0]
y = data[:,1]
m = len(t)  # Número de puntos de datos
xopt = ovo_algorithm(t, y)
print("Parámetros óptimos:", xopt)
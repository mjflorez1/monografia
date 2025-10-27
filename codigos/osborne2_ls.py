import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

size_img = 0.6
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = [size_img * 6.4,size_img * 4.8]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def model(t, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return (x1 * np.exp(-t * x5)
          + x2 * np.exp(-x6 * (t - x9) ** 2)
          + x3 * np.exp(-x7 * (t - x10) ** 2)
          + x4 * np.exp(-x8 * (t - x11) ** 2))

# Cargar datos
data = np.loadtxt('txt/data_osborne2.txt')
t = data[:, 0]
y = data[:, 1]

# Vector base (punto inicial)
x_star = [1.3, 0.65, 0.65, 0.7, 0.6, 3, 5, 7, 2, 4.5, 5.5]

# Modelo evaluado en el punto inicial
w = model(t, *x_star)

# Función objetivo
def objetivo(x):
    y_modelo = model(t, *x)
    return 0.5 * np.sum((y - y_modelo) ** 2)

# Punto inicial (mismo que x_star)
x0 = x_star
bounds = [(-20, 20)] * 11

# Minimización
res = minimize(objetivo, x0, method='L-BFGS-B', bounds=bounds)

# Guardar solución
np.savetxt("txt/sol_osborne2_ls.txt", res.x, fmt="%.6f")
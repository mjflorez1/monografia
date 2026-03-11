import numpy as np
import matplotlib.pyplot as plt

size_img = 0.6
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = [size_img * 6.4, size_img * 4.8]
plt.rc('font', family='serif')

# Datos
x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
y = [0.01140, 0.00612, 0.00863, 0.00679, 0.00645, 0.00001, 0.00012, 0.00016, 0.00002]

# Gráfica
plt.figure()
plt.plot(x, y, 'o-', markersize = 3)   # círculos + línea
plt.yscale('log')

# Etiquetas
plt.xlabel('Número de outliers (o)')
plt.ylabel(r'$f(x^*)$ (log scale)')

plt.show()
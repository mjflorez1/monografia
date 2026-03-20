import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Dominio
x = np.linspace(0.5, 4.5, 1000)

# Funciones
f = 0.5 * np.sin((1/5) * x) + 0.3
g = np.sin((2/3) * x) + 0.4
h = (2/3) * np.exp(x/5) - 0.2
p = 0.5 * np.cos(x) + 0.9
q = 0.5 * np.exp(x/3) - 0.5
r = 0.5 * np.cos(x/2) + 0.8

F = np.vstack([f, g, h, p, q, r])
F_sorted = np.sort(F, axis=0)

# p = 3 → índice 2
f_ovo = F_sorted[3]

delta = 0.2

# Límites de la región
lower = f_ovo - delta
upper = f_ovo + delta

# Figura
fig, ax = plt.subplots()

# Graficar funciones
ax.plot(x, f)
ax.plot(x, g)
ax.plot(x, h)
ax.plot(x, p)
ax.plot(x, q)
ax.plot(x, r)

# Elementos dinámicos
region = None
point, = ax.plot([], [], 'ro')

ax.set_xlim(0.5, 4.5)
ax.set_ylim(np.min(F)-0.5, np.max(F)+0.5)

# Inicialización
def init():
    point.set_data([], [])
    return point,

# Animación
def update(i):
    global region
    
    # eliminar región anterior
    if region is not None:
        region.remove()
    
    # dibujar región acumulada hasta i
    region = ax.fill_between(x[:i], lower[:i], upper[:i],
                             color='purple', alpha=0.2)
    
    # punto actual
    point.set_data([x[i]], [f_ovo[i]])
    
    return point, region

ani = FuncAnimation(fig, update, frames=len(x),
                    init_func=init, interval=20, blit=False)

plt.show()
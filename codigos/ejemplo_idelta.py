import numpy as np
import matplotlib.pyplot as plt

# Dominio
x = np.linspace(0.5, 4.5, 300)

# Funciones
f = 0.5 * np.sin((1/5) * x) + 0.3
g = np.sin((2/3) * x) + 0.7
h = (2/3) * np.exp(x/5) - 0.2
p = 0.5 * np.cos(x) + 1
q = 0.5 * np.exp(x/3) - 0.5

F = np.vstack([f, g, h, p, q])
F_sorted = np.sort(F, axis=0)

f_ovo = F_sorted[2]
delta = 0.3

lower = f_ovo - delta
upper = f_ovo + delta

# Generar frames
for i in range(0, len(x), 5):  # salto para no generar demasiadas imágenes
    fig, ax = plt.subplots()
    
    ax.plot(x, f)
    ax.plot(x, g)
    ax.plot(x, h)
    ax.plot(x, p)
    ax.plot(x, q)
    
    ax.fill_between(x[:i], lower[:i], upper[:i],
                    color='purple', alpha=0.2)
    
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(np.min(F)-0.5, np.max(F)+0.5)
    
    plt.savefig(f"frame_{i}.png")
    plt.close()
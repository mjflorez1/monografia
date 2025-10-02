import numpy as np
import matplotlib.pyplot as plt

n = 5
m = 33

y = np.array([
    0.844, 0.908, 0.932, 0.936, 0.925, 0.908, 0.881, 0.850, 0.818,
    0.784, 0.751, 0.718, 0.685, 0.658, 0.628, 0.603, 0.580, 0.558,
    0.538, 0.522, 0.506, 0.490, 0.478, 0.467, 0.457, 0.448, 0.438,
    0.431, 0.424, 0.420, 0.414, 0.411, 0.406
])

noise = 0.2
y[10] += noise
y[11] += noise
y[12] += noise
y[13] += noise
y[14] += noise

t = np.array([10*(i-1) for i in range(1, m+1)])

with open("data_osborne1.txt", "w") as f:
    for i in range(m):
        f.write(f'{t[i]} {y[i]}\n')


plt.plot(t,y,"o")
plt.savefig("figuras/osborne.pdf",bbox_inches = "tight")
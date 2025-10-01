import numpy as np
import matplotlib.pyplot as plt

n = 3
m = 15

u = np.arange(1, m+1)
v = 16 - u
w = np.minimum(u, v)

y = np.array([0.14, 0.18, 0.22, 0.25, 0.29,
              0.32, 0.35, 0.39, 0.37, 0.58,
              0.73, 0.96, 1.34, 2.10, 4.39])

with open("bard_data.txt", "w") as f:
    for i in range(m):
        f.write(f"{i+1} {u[i]} {v[i]} {w[i]} {y[i]}\n")
        
plt.plot(u,y,"o")
plt.show()
import numpy as np
import matplotlib.pyplot as plt

n = 3
m = 15

t = np.array([(8 - i)/2 for i in range(1,m+1)])
    
y = np.array([0.0009, 0.0044, 0.0175, 0.0540, 0.1295, 0.2420, 0.3521, 0.3989, 0.3521, 0.2420, 0.1295,
              0.0540, 0.0175, 0.0044, 0.0009])

noise = 0.5
y[4] += noise
y[5] += noise
y[6] += noise

with open("data_gauss.txt", "w") as f:
    for i in range(m):
        f.write(f"{t[i]} {y[i]}\n")

plt.plot(t,y,"o")
plt.savefig("figuras/gauss.pdf", bbox_inches="tight")
plt.show()
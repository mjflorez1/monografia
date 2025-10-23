import numpy as np
import matplotlib.pyplot as plt
import random

size_img = 0.6
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = [size_img * 6.4,size_img * 4.8]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Definimos el modelo cubico
def model(t,x1,x2,x3,x4):
    return x1 + x2 * t + x3 * t**2 + x4 * t**3

#Definimos los valores de t_i (de 1 a 46)
m = 46
t = np.linspace(-1,3.5,m)
xstar = np.array([0, 2, -3, 1])

y = 10 * np.ones(m)
random.seed(1234)

for i in range(6):
    y[i] = model(t[i],*xstar) + random.uniform(-0.01, 0.01)

for i in range(16,m):
    y[i] = model(t[i],*xstar) + random.uniform(-0.01, 0.01)

with open("txt/data.txt","w") as f:    
    for i in range(m):
        f.write(f"{t[i]} {y[i]}\n")
        
plt.plot(t,y,"o",ms=3)
plt.savefig("figuras/cubic.pdf", bbox_inches="tight")
plt.show()
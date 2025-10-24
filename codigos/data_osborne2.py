import numpy as np
import matplotlib.pyplot as plt

size_img = 0.6
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = [size_img * 6.4,size_img * 4.8]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

n = 11
m = 78

y = np.array([
    1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786, 0.725,
    0.746, 0.679, 0.608, 0.655, 0.616, 0.606, 0.602, 0.626, 0.651, 0.724,
    0.649, 0.649, 0.694, 0.644, 0.624, 0.661, 0.612, 0.558, 0.533, 0.495,
    0.500, 0.423, 0.395, 0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429,
    0.523, 0.562, 0.607, 0.653, 0.672, 0.708, 0.633, 0.668, 0.645, 0.632,
    0.591, 0.559, 0.597, 0.625, 0.739, 0.710, 0.729, 0.720, 0.636, 0.581,
    0.428, 0.292, 0.162, 0.098, 0.054
])

y = np.insert(y,4,1.418)
y = np.insert(y,9,1.146)
y = np.insert(y,15,1.132)
y = np.insert(y,20,1.071)
y = np.insert(y,22,1.175)
y = np.insert(y,24,1.266)
y = np.insert(y,33,0.979)
y = np.insert(y,40,0.867)
y = np.insert(y,42,1.109)
y = np.insert(y,46,1.211)
y = np.insert(y,47,0.942)
y = np.insert(y,67,1.338)
y = np.insert(y,70,1.148)

t = np.array([(i-1)/10 for i in range(1, m+1)])

with open("txt/data_osborne2.txt", "w") as f:
    for i in range(m):
        f.write(f'{t[i]} {y[i]}\n')


plt.plot(t,y,"o",ms=3)
plt.savefig("figuras/osborne2.pdf",bbox_inches = "tight")
plt.show()
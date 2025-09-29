import numpy as np
import matplotlib.pyplot as plt

# Dimensiones
n = 3   # número de parámetros
m = 16  # número de datos

# Generar valores de t_i
t = np.array([45 + 5*i for i in range(1, m+1)])

# Valores observados y_i
y = np.array([
    34780, 28610, 23650, 19630, 16370, 13720, 11540, 9744,
    8261, 7030, 6005, 5147, 4427, 3820, 3307, 2872
])

with open("meyer_data.txt", "w") as f:
    for i in range(m):
        f.write(f"{t[i]} {y[i]}\n")

plt.plot(t,y,"o")
plt.show()
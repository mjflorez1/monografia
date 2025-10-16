import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def f1(x1):
    x2 = 1 - x1
    return x1**2 + x2**2

# Restricción: x1 >= 1/3
res = minimize_scalar(f1, bounds=(1/3, 1), method="bounded")

x1_opt = res.x
x2_opt = 1 - x1_opt
f_opt = res.fun

print("===== RESULTADOS =====")
print(f"x1* = {x1_opt:.6f}")
print(f"x2* = {x2_opt:.6f}")
print(f"Valor óptimo f(x*) = {f_opt:.6f}")
print("======================")


x1_vals = np.linspace(0, 1, 200)
x2_vals = 1 - x1_vals

X, Y = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
Z = X**2 + Y**2

plt.figure(figsize=(8,5))

plt.fill_between(x1_vals, x2_vals, where=(x1_vals >= 1/3),
                 color="lightblue", alpha=0.5, label="Región factible")

plt.plot(x1_vals, x2_vals, "b-", label="x1 + x2 = 1")

contours = plt.contour(X, Y, Z, levels=[0.5, 0.7, 1.0], colors="gray")
plt.clabel(contours, inline=True, fontsize=10)

plt.plot(x1_opt, x2_opt, "ro", markersize=10,
         label=f"Óptimo: ({x1_opt:.3f}, {x2_opt:.3f})")

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.show()
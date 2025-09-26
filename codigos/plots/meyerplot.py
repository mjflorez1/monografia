import numpy as np
import matplotlib.pyplot as plt

# Datos
y_data = np.array([34780, 28610, 23650, 19630, 16370, 13720, 11540, 9744,
                   8261, 7030, 6005, 5147, 4427, 3820, 3307, 2872])
t_data = 45 + 5 * np.arange(1, 17)

# Función modelo
def g(t, x):
    return x[0] * np.exp(x[1] / (t + x[2]))

# Función residual
def f_residuals(x, t, y):
    return g(t, x) - y

# Parámetro inicial
x0 = np.array([0.02, 4000, 250])

# Curva modelo
t = np.linspace(50, 200, 300)
y_model = g(t, x0)

# Residuos en los datos
residuos = f_residuals(x0, t_data, y_data)

# Graficar
plt.figure(figsize=(10,5))

# Modelo continuo
plt.subplot(1,2,1)
plt.plot(t, y_model, 'b-', label="Modelo $g(t;x_0)$")
plt.scatter(t_data, y_data, c="red", label="Datos $y_i$")
plt.xlabel("t")
plt.ylabel("y")
plt.title("Modelo vs Datos")
plt.legend()

# Residuos
plt.subplot(1,2,2)
plt.stem(t_data, residuos, basefmt="k")
plt.axhline(0, color="k", linestyle="--")
plt.xlabel("t")
plt.ylabel("Residual $f_i(x_0)$")
plt.title("Residuos del modelo")

plt.tight_layout()
plt.show()
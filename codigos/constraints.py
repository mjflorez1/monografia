import numpy as np

def load_data(filename="data.txt"):
    with open(filename, 'r') as f:
        lines = f.readlines()
    m = int(lines[0])
    t, y = [], []
    for line in lines[1:]:
        ti, yi = map(float, line.strip().split())
        t.append(ti)
        y.append(yi)
    return np.array(t), np.array(y)

# Modelo cúbico y gradiente
def model(t, x):
    return x[0] + x[1]*t + x[2]*t**2 + x[3]*t**3

def f_i(x, t_i, y_i):
    return 0.5 * (model(t_i, x) - y_i)**2

def grad_f_i(x, t_i, y_i):
    diff = model(t_i, x) - y_i
    return diff * np.array([1, t_i, t_i**2, t_i**3])

# Calcular I_delta(x)
def compute_I_delta(x, t, y, q, delta):
    f_vals = np.array([f_i(x, t[i], y[i]) for i in range(len(t))])
    indices = np.argsort(f_vals)
    f_q = f_vals[indices[q-1]]
    I_delta = [i for i in range(len(f_vals)) if abs(f_vals[i] - f_q) <= delta]
    return I_delta

# Guardar restricciones
def guardar_restricciones_txt(x, t, y, I_delta, filename="constraints.txt"):
    with open(filename, "w") as f:
        for j in I_delta:
            grad = grad_f_i(x, t[j], y[j])
            restr = " + ".join([f"{grad[k]:.4f}*x{k+1}" for k in range(len(grad))])
            f.write(f"Restricción {j}: {restr} <= w\n")

def main():
    t, y = load_data("data.txt")
    x = np.array([-1.0, -2.0, 1.0, -1.0])  # punto de evaluación
    q = 32
    delta = 0.1

    I_delta = compute_I_delta(x, t, y, q, delta)
    guardar_restricciones_txt(x, t, y, I_delta)

    print(f"Se guardaron {len(I_delta)} restricciones en 'constraints.txt'")

if __name__ == "__main__":
    main()
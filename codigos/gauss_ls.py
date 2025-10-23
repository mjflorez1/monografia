import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def model(t, x1, x2, x3):
    return x1 * np.exp((-x2 * ((t - x3)**2)) / 2)

data = np.loadtxt('txt/data_gauss.txt')
t = data[:, 0]
y = data[:, 1]

x_star = [0.4, 1.0, 0.0]
w = x_star[0] + np.exp((-x_star[1] * ((t - x_star[2]) ** 2)) / 2)

def objetivo(x):
    y_modelo = model(t, *x)
    return 0.5 * np.sum((y - y_modelo) ** 2)

x0 = [1, 1, 0]
bounds = [(-10,10)] * 3
res = minimize(objetivo, x0, method='L-BFGS-B', bounds=bounds)

np.savetxt("txt/sol_gauss_ls.txt", res.x, fmt="%.6f")
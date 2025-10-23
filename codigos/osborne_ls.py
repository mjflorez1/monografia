import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def model(t, x1, x2, x3, x4, x5):
    return x1 + (x2 * np.exp(-t * x4)) + (x3 * np.exp(-t * x5))

data = np.loadtxt("data_osborne1.txt")
t = data[:,0]
y = data[:,1]

# MÉTODO minimize
x_star = [0.5, 1.5, -1.0, 0.01, 0.02]
w = x_star[0] + (x_star[1] * np.exp(-t * x_star[3])) + (x_star[2] * np.exp(-t * x_star[4]))

def objetivo(x):
    y_modelo = model(t, *x)
    return 0.5 * np.sum((y - y_modelo)**2)

x0 = [0.5, 1.5, -1.0, 0.01, 0.02]
bounds = [(-1, 1)] * 5
res = minimize(objetivo, x0, method='L-BFGS-B', bounds=bounds)

np.savetxt("sol_osborne_ls.txt", res.x, fmt="%.6f")  
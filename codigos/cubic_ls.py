import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def model(t,x1,x2,x3,x4):
    res = x1 + (x2 * t) + x3 * (t**2) + x4 * (t**3)
    return res

data = np.loadtxt("data.txt")
t = data[:,0]
y = data[:,1]

# MÉTODO minimize
x_star = [0, 2, -3, 1]
w = x_star[0] + x_star[1]*t + x_star[2]*t**2 + x_star[3]*t**3

def objetivo(x):
    y_modelo = np.polyval(x[::-1], t)
    return 0.5 * np.sum((y - y_modelo) ** 2)

x0 = [-1, -2, 1, -1]
bounds = [(-10, 10)] * 4
res = minimize(objetivo, x0, method='L-BFGS-B', bounds=bounds)

np.savetxt("sol_cubic_ls.txt", res.x, fmt="%.6f")  



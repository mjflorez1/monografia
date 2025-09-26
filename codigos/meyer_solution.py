import numpy as np
import matplotlib.pyplot as plt

def model(t,x0,x1,x2):
    return x0 * np.exp(x1 / (t + x2))

data = np.loadtxt("meyer_data.txt")
xstar = np.array([0.02, 4000, 250])

t = np.linspace(data[0,0],data[-1,0],100)

plt.plot(data[:,0],data[:,1],"o")
plt.plot(t,model(t,*xstar))
plt.show()
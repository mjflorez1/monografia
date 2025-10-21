import numpy as np
import matplotlib.pyplot as plt

def model(t,x1,x2,x3,x4):
    res = x1 + (x2 * t) + x3 * (t**2) + x4 * (t**3)
    return res

size_img = 0.6
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = [size_img * 6.4,size_img * 4.8]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sol_ls = np.loadtxt("sol_cubic_ls.txt") 
data = np.loadtxt("data.txt")

t = np.linspace(data[:,0][0],data[:,0][-1],1000)
plt.plot(t,model(t,*sol_ls),lw=1)
plt.plot(data[:,0],data[:,1],"ok",ms=3)
plt.savefig("figuras/cubic_cauchy.pdf",bbox_inches="tight")



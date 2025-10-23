import numpy as np
import matplotlib.pyplot as plt

def model(t, x0, x1, x2):
    return x0 * np.exp((-x1 * ((t - x2)**2)) / 2)

size_img = 0.6
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = [size_img * 6.4,size_img * 4.8]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sol_0 = np.loadtxt("txt/gauss_cn/outs_gauss_cn0.txt")
sol_1 = np.loadtxt("txt/gauss_cn/outs_gauss_cn1.txt")
sol_2 = np.loadtxt("txt/gauss_cn/outs_gauss_cn2.txt")
sol_3 = np.loadtxt("txt/gauss_cn/outs_gauss_cn3.txt")
data = np.loadtxt("txt/data_gauss.txt")

t = np.linspace(data[:,0][0],data[:,0][-1],1000)
plt.plot(t,model(t,*sol_0),lw=1,label="o = 0")
plt.plot(t,model(t,*sol_1),lw=1,label="o = 1")
plt.plot(t,model(t,*sol_2),lw=1,label="o = 2")
plt.plot(t,model(t,*sol_3),lw=1,label="o = 3")
plt.plot(data[:,0],data[:,1],"ok",ms=3)
plt.legend(fontsize=6,loc="upper left")
plt.savefig("figuras/gauss_cn_outs.pdf",bbox_inches="tight")
plt.show()


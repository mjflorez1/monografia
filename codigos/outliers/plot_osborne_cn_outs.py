import numpy as np
import matplotlib.pyplot as plt

def model(t, x0, x1, x2, x3, x4):
    return x0 + (x1 * np.exp(-t * x3)) + (x2 * np.exp(-t * x4))

size_img = 0.6
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = [size_img * 6.4,size_img * 4.8]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sol_0 = np.loadtxt("txt/osborne_cn/outs_osborne_cn0.txt")
sol_1 = np.loadtxt("txt/osborne_cn/outs_osborne_cn1.txt")
sol_2 = np.loadtxt("txt/osborne_cn/outs_osborne_cn2.txt")
sol_3 = np.loadtxt("txt/osborne_cn/outs_osborne_cn3.txt")
sol_4 = np.loadtxt("txt/osborne_cn/outs_osborne_cn4.txt")
sol_5 = np.loadtxt("txt/osborne_cn/outs_osborne_cn5.txt")
data = np.loadtxt("txt/data_osborne1.txt")

t = np.linspace(data[:,0][0],data[:,0][-1],1000)
plt.plot(t,model(t,*sol_0),lw=1,label="o = 0")
plt.plot(t,model(t,*sol_1),lw=1,label="o = 1")
plt.plot(t,model(t,*sol_2),lw=1,label="o = 2")
plt.plot(t,model(t,*sol_3),lw=1,label="o = 3")
plt.plot(t,model(t,*sol_4),lw=1,label="o = 4")
plt.plot(t,model(t,*sol_5),lw=1,label="o = 5")
plt.plot(data[:,0],data[:,1],"ok",ms=3)
plt.legend(fontsize=6,loc="lower left")
plt.savefig("figuras/osborne_cn_outs.pdf",bbox_inches="tight")
plt.show()

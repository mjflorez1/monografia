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

sol_0 = np.loadtxt("txt/cubic_cauchy/out_cubic_0.txt")
sol_1 = np.loadtxt("txt/cubic_cauchy/out_cubic_1.txt")
sol_2 = np.loadtxt("txt/cubic_cauchy/out_cubic_2.txt")
sol_3 = np.loadtxt("txt/cubic_cauchy/out_cubic_3.txt")
sol_4 = np.loadtxt("txt/cubic_cauchy/out_cubic_4.txt")
sol_5 = np.loadtxt("txt/cubic_cauchy/out_cubic_5.txt")
sol_6 = np.loadtxt("txt/cubic_cauchy/out_cubic_6.txt")
sol_7 = np.loadtxt("txt/cubic_cauchy/out_cubic_7.txt")
sol_8 = np.loadtxt("txt/cubic_cauchy/out_cubic_8.txt")
sol_9 = np.loadtxt("txt/cubic_cauchy/out_cubic_9.txt")
sol_10 = np.loadtxt("txt/cubic_cauchy/out_cubic_10.txt")
#sol_cauchy = np.loadtxt("txt/sol_cubic_cauchy.txt")
data = np.loadtxt("txt/data.txt")

t = np.linspace(data[:,0][0],data[:,0][-1],1000)
plt.plot(t,model(t,*sol_0),lw=1,label="o = 0")
plt.plot(t,model(t,*sol_1),lw=1,label="o = 1")
plt.plot(t,model(t,*sol_2),lw=1,label="o = 2")
plt.plot(t,model(t,*sol_3),lw=1,label="o = 3")
plt.plot(t,model(t,*sol_4),lw=1,label="o = 4")
plt.plot(t,model(t,*sol_5),lw=1,label="o = 5")
plt.plot(t,model(t,*sol_6),lw=1,label="o = 6")
plt.plot(t,model(t,*sol_7),lw=1,label="o = 7")
plt.plot(t,model(t,*sol_8),lw=1,label="o = 8")
plt.plot(t,model(t,*sol_9),lw=1,label="o = 9")
plt.plot(t,model(t,*sol_10),lw=1,label="o = 10")
#plt.plot(t,model(t,*sol_cauchy),lw=1,label="OVO tipo Cauchy")
plt.plot(data[:,0],data[:,1],"ok",ms=3)
plt.legend(fontsize=4,loc="upper left")
plt.savefig("figuras/cubic_cauchy_outs.pdf",bbox_inches="tight")
plt.show()



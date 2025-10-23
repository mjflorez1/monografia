import numpy as np
import matplotlib.pyplot as plt

def model(t, x0, x1, x2, x3, x4):
    return x0 + (x1 * np.exp(-t * x3)) + (x2 * np.exp(-t * x4))

size_img = 0.6
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = [size_img * 6.4,size_img * 4.8]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sol_ls = np.loadtxt("txt/sol_osborne_ls.txt")
sol_cauchy = np.loadtxt("txt/sol_osborne_cn.txt")
data = np.loadtxt("txt/data_osborne1.txt")

t = np.linspace(data[:,0][0],data[:,0][-1],1000)
plt.plot(t,model(t,*sol_ls),lw=1,label='OLS')
plt.plot(t,model(t,*sol_cauchy),lw=1,label='OVO tipo Cuasi-Newton')
plt.plot(data[:,0],data[:,1],"ok",ms=3,label='Datos')
plt.savefig("figuras/osborne_cn.pdf",bbox_inches="tight")
plt.legend(fontsize=6,loc='best')
plt.show()
import numpy as np
import matplotlib.pyplot as plt

def model(t, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    return (x1 * np.exp(-t * x5) + 
            x2 * np.exp(-x6 * (t - x9)  ** 2) + 
            x3 * np.exp(-x7 * (t - x10) ** 2) +
            x4 * np.exp(-x8 * (t - x11) ** 2))

size_img = 0.6
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = [size_img * 6.4,size_img * 4.8]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sol_ls = np.loadtxt("txt/sol_osborne2_ls.txt")
sol_cauchy = np.loadtxt("txt/sol_osborne2_cauchy.txt")
data = np.loadtxt("txt/data_osborne2.txt")

t = np.linspace(data[:,0][0],data[:,0][-1],1000)
plt.plot(t,model(t,*sol_ls),lw=1,label='OLS')
plt.plot(t,model(t,*sol_cauchy),lw=1,label='OVO tipo Cauchy')
plt.plot(data[:,0],data[:,1],"ok",ms=3,label='Datos')
plt.legend(fontsize=6,loc='best')
plt.savefig("figuras/osborne2_cauchy.pdf",bbox_inches="tight")
plt.show()
import numpy as np
import matplotlib.pyplot as plt

size_img = 0.6
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = [size_img * 6.4,size_img * 4.8]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

x = np.linspace(-5,5,1000)
f1 = x
f2 = -x
f3 = np.abs(x)

plt.plot(x, f1, label=r"$f_1(x)=x$")
plt.plot(x, f2, label=r"$f_2(x)=-x$")
plt.plot(x, f3, linewidth=3, color='black', label=r"$f_{ovo}(x)=|x|$")
plt.legend(fontsize=8, loc="best")
plt.savefig('figuras/ejemplo.pdf',bbox_inches='tight')
plt.show()
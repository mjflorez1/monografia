import matplotlib.pyplot as plt
import numpy as np

size_img = 0.6
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = [size_img * 6.4,size_img * 4.8]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

o = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
f = np.array([0.004810, 0.004794376885243772, 0.004495042149349769, 0.002575069342581464,
              0.001246774312192673, 5.00311575328171e-06, 1.9988682036563517e-06,
              2.74908474489713e-06, 2.732695004757646e-06])

plt.plot(o, f, 'o-', linewidth=1, markersize=3)
plt.yscale('log')
plt.xlabel('Número de outliers ($o$)', fontsize=12)
plt.ylabel('$f(x^{*})$', fontsize=12)
plt.savefig("figuras/osborne_cn_vs_outliers.pdf", bbox_inches = 'tight')
plt.show()
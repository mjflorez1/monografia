import matplotlib.pyplot as plt
import numpy as np

size_img = 0.6
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = [size_img * 6.4,size_img * 4.8]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

o = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
f = np.array([0.004715692707991851, 0.004633830707903854, 0.004294181811584838, 0.004600617739487099,
              0.003985378664528316, 4.1413632131026687e-05, 2.343204215979604e-05, 3.338280347742875e-05,
              3.7116702771239276e-05])

plt.plot(o, f, 'o-', linewidth=1, markersize=3)
plt.yscale('log')
plt.xlabel('Número de outliers ($o$)', fontsize=12)
plt.ylabel('$f(x^{*})$', fontsize=12)
plt.savefig("figuras/osborne_cn_vs_outliers.pdf", bbox_inches = 'tight')
plt.show()
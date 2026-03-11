import matplotlib.pyplot as plt
import matplotlib.patches as patches

size_img = 0.6
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = [size_img * 6.4,size_img * 4.8]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Datos
arr = [5, 8, 6, 1, 12, 3]

fig, ax = plt.subplots()

# Dibujar celdas
for i, v in enumerate(arr):
    color = "#f28c28" if i == 0 else "#e6e66a"  # primera diferente
    rect = patches.Rectangle((i, 1), 1, 1, edgecolor="black", facecolor=color)
    ax.add_patch(rect)
    ax.text(i + 0.5, 1.5, str(v), ha='center', va='center', fontsize=14)

# Flecha hacia abajo
ax.annotate(
    "",
    xy=(0.5, 0.2),
    xytext=(0.5, 1),
    arrowprops=dict(arrowstyle="->", linewidth=2)
)

# Rectángulo inferior
rect2 = patches.Rectangle((-0.5, -0.5), 6.8, 0.7, edgecolor="black", facecolor="#cfcfcf")
ax.add_patch(rect2)

ax.set_xlim(-0.5, len(arr) + 0.5)
ax.set_ylim(-1, 2.5)
ax.axis("off")

plt.savefig("figuras/imagen1.pdf",bbox_inches="tight")
plt.show()
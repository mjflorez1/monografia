import matplotlib.pyplot as plt
import matplotlib.patches as patches

size_img = 0.6
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = [size_img * 6.4,size_img * 4.8]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

arr = [8, 6, 1, 12, 3]
insertado = 5

fig, ax = plt.subplots()

# ---- fila superior ----
for i, v in enumerate(arr):
    color = "#f28c28" if i == 0 else "#e6e66a"
    r = patches.Rectangle((i+2, 2), 1, 1, edgecolor="black", facecolor=color, linewidth=1.5)
    ax.add_patch(r)
    ax.text(i+2.5, 2.5, str(v), ha="center", va="center", fontsize=16)

# ---- caja grande inferior ----
base = patches.Rectangle((0, 0), 7, 1, edgecolor="black", facecolor="#cfcfcf", linewidth=1.5)
ax.add_patch(base)

# ---- primer elemento insertado ----
r = patches.Rectangle((0, 0), 1, 1, edgecolor="black", facecolor="#f2ef2e", linewidth=1.5)
ax.add_patch(r)
ax.text(0.5, 0.5, str(insertado), ha="center", va="center", fontsize=16)

# ---- flecha ----
ax.annotate(
    "",
    xy=(0.5, 1),
    xytext=(0.5, 1.8),
    arrowprops=dict(arrowstyle="simple", color="black")
)

ax.set_xlim(-0.5, 8)
ax.set_ylim(-0.5, 3.5)
ax.axis("off")

plt.savefig("figuras/imagen2.pdf",bbox_inches="tight")
plt.show()
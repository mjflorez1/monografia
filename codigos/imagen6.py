import matplotlib.pyplot as plt
import matplotlib.patches as patches

size_img = 0.6
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = [size_img * 6.4, size_img * 4.8]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

top = [1, 12, 3]
bottom = [5, 6, 8]

fig, ax = plt.subplots()

# ----- fila superior -----
start_x = 3
for i, v in enumerate(top):
    color = "#f28c28" if i == 0 else "#e6e66a"
    rect = patches.Rectangle((start_x + i, 2), 1, 1,
                             edgecolor="black",
                             facecolor=color,
                             linewidth=1.5)
    ax.add_patch(rect)

    ax.text(start_x + i + 0.5, 2.5, str(v),
            ha="center", va="center", fontsize=18)

# ----- caja base -----
base = patches.Rectangle((0, 0), 7, 1,
                         edgecolor="black",
                         facecolor="#cfcfcf",
                         linewidth=1.5)
ax.add_patch(base)

# ----- elementos colocados -----
for i, v in enumerate(bottom):
    rect = patches.Rectangle((i, 0), 1, 1,
                             edgecolor="black",
                             facecolor="#f3f02f",
                             linewidth=1.5)
    ax.add_patch(rect)

    ax.text(i + 0.5, 0.5, str(v),
            ha="center", va="center", fontsize=18)

# ----- flecha vertical (1 -> 5) -----
ax.annotate(
    "",
    xy=(0.5, 1),
    xytext=(0.5, 1.8),
    arrowprops=dict(arrowstyle="->", linewidth=2)
)

ax.set_xlim(-0.5, 8)
ax.set_ylim(-0.5, 3.5)
ax.axis("off")

plt.savefig("figuras/imagen6.pdf", bbox_inches="tight")
plt.show()
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch

size_img = 0.6
plt.rcParams.update({'font.size': 11})
plt.rcParams['figure.figsize'] = [size_img * 6.4, size_img * 4.8]
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

top = [6, 1, 12, 3]
bottom = [5, 8]

fig, ax = plt.subplots()

# ----- fila superior -----
start_x = 2
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

# ----- flecha diagonal (6 -> 8) -----
ax.annotate(
    "",
    xy=(1.5, 1),
    xytext=(2.5, 2),
    arrowprops=dict(arrowstyle="->", linewidth=2)
)

# ===== CURVA CON PUNTA VISIBLE =====

start = (1.5, 0.0)     # borde inferior central del 8
end   = (2.5, 0.5)     # centro del espacio vacío

control1 = (1.5, -1.0)
control2 = (2.5, -1.0)

path = Path(
    [start, control1, control2, end],
    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
)

arrow = FancyArrowPatch(
    path=path,
    arrowstyle='-|>',      # punta clara
    mutation_scale=18,     # tamaño de la punta
    linewidth=2,
    color='black',
    shrinkA=0,
    shrinkB=0
)

ax.add_patch(arrow)

ax.set_xlim(-0.5, 8)
ax.set_ylim(-1.5, 3.5)
ax.axis("off")

plt.savefig("figuras/imagen5.pdf", bbox_inches="tight")
plt.show()
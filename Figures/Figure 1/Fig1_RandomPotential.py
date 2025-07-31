import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import rc
import cmasher as cmr

# Plot settings
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "lines.linewidth": 1,
    "grid.alpha": 0.3,
    "axes.grid": True,
})
rc('text.latex', preamble=r'\usepackage{amsmath}')  # Allows using AMS math symbols
def constructPotential(size, mean=0, stdev=1):
    V = np.zeros((2*size+1, 2*size+1), dtype=complex)
    for i in range(size + 1):
        for j in range(-size, size + 1):
            real_part = np.random.normal(0, stdev)
            imag_part = np.random.normal(0, stdev)
            V[size + i, size + j] = real_part + 1j * imag_part
            if not (i == 0 and j == 0):  
                V[size - i, size - j] = real_part - 1j * imag_part
    V[size, size] = 0  # Set DC offset to zero
    return V / np.sqrt(NUM_STATES)  # Normalized potential


def plotRandomPotential3D(inverse_potential, ax, title=""):
    shifted = np.fft.ifftshift(inverse_potential)
    real = np.fft.ifft2(shifted).real

    x = np.arange(real.shape[1])
    y = np.arange(real.shape[0])
    print(real.shape[1])
    print(real.shape[1])
    X, Y = np.meshgrid(x, y)
    Z = real
    surf = ax.plot_surface(X, Y, Z, cmap="magma", edgecolor="none", alpha=0.9, )
    # ax.plot_surface(X, Y, np.zeros_like(Z), edgecolor='k', linewidth=2, alpha=0, rstride=100, cstride=100,zorder=2)

    ax.zaxis.set_label_coords(-0.13, 0.5)    # manual label shift
    for t in ax.get_zticklabels():           # manual tick-label shift
        t.set_horizontalalignment('left')
        t.set_x(-0.13)

    ax.set_xlabel(r"$x$", labelpad=-15)
    ax.set_ylabel(r"$y$", labelpad=-15)
    ax.set_zlabel(r"$V(\mathbf{r})$", labelpad=-15)
    # ax.set_zlabel("Magnitude of Potential", labelpad=-85)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # ax.set_proj_type('ortho')

    # Add boundary at z=0
    return surf

NUM_STATES = 32
POTENTIAL_MATRIX_SIZE = int(4*np.sqrt(NUM_STATES))

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(3.4, 3))
V_2D = constructPotential(POTENTIAL_MATRIX_SIZE)

# 3D plot
ax2 = fig.add_subplot(111, projection='3d')
surf = plotRandomPotential3D(V_2D, ax2)

# Colorbar
cbar = fig.colorbar(surf, ax= ax2, orientation="vertical",location="left",
                    label="$V(\mathbf{r})$ (color scale)", shrink=0.7, pad=0.08)
cbar.ax.tick_params(labelsize=8)
cbar.ax.yaxis.set_label_position('right')

plt.tight_layout()
# plt.show()
# plt.savefig("testingV2.pdf")
plt.savefig("Figures/Figure1v1.3.pdf", bbox_inches='tight')  # Minimize extra space

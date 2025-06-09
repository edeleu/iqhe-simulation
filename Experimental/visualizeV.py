import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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

def plotRandomPotential2D(inverse_potential, ax, title=""):
    # Convert to real-space potential
    shifted = np.fft.ifftshift(inverse_potential)
    real = np.fft.ifft2(shifted).real

    # Grid coordinates for pixel edges
    ny, nx = real.shape
    x = np.arange(nx + 1)
    y = np.arange(ny + 1)

    # Plot using pcolormesh (vector graphics)
    im = ax.pcolormesh(x, y, real, cmap='inferno', shading='auto')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)

    # ax.set_title(title)

    return im

# def plotRandomPotential2D(inverse_potential, ax, title=""):
#     shifted = np.fft.ifftshift(inverse_potential)
#     real = np.fft.ifft2(shifted).real
#     im = ax.imshow(real, extent=[0, real.shape[1], 0, real.shape[0]], origin='lower', cmap='inferno',rasterized=False)
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_xticks([])
#     ax.set_yticks([])
#     return im

def plotRandomPotential3D(inverse_potential, ax, title=""):
    shifted = np.fft.ifftshift(inverse_potential)
    real = np.fft.ifft2(shifted).real

    x = np.arange(real.shape[1])
    y = np.arange(real.shape[0])
    print(real.shape[1])
    print(real.shape[1])
    X, Y = np.meshgrid(x, y)
    Z = real
    surf = ax.plot_surface(X, Y, Z, cmap="inferno", edgecolor="none", alpha=0.9)
    # ax.plot_surface(X, Y, np.zeros_like(Z), edgecolor='k', linewidth=2, alpha=0, rstride=100, cstride=100,zorder=2)

    ax.set_xlabel("x", labelpad=-15)
    ax.set_ylabel("y", labelpad=-15)
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

fig = plt.figure(figsize=(6.8, 3))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05)  # tighter space

# 2D plot
ax1 = fig.add_subplot(gs[0])
V_2D = constructPotential(POTENTIAL_MATRIX_SIZE)
plotRandomPotential2D(V_2D, ax1)

# 3D plot
ax2 = fig.add_subplot(gs[1], projection='3d')
surf = plotRandomPotential3D(V_2D, ax2)

# Colorbar
cbar = fig.colorbar(surf, ax=[ax1, ax2], orientation="vertical",
                    label="Magnitude of Potential", shrink=0.75, pad=0.04)
cbar.ax.tick_params(labelsize=8)
cbar.ax.yaxis.set_label_position('left')

plt.tight_layout()
# plt.savefig("testingV2.pdf")
plt.savefig("testingV.pdf", bbox_inches='tight')  # Minimize extra space

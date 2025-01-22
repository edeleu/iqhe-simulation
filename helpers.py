import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib import rc

# Set global matplotlib style configurations
plt.rcParams.update({
    "text.usetex": True,                   # Use LaTeX for text rendering
    "font.family": "serif",                # Use a serif font (e.g., Times New Roman)
    "font.serif": ["Computer Modern"],     # Specify Times New Roman
    "axes.titlesize": 10,                  # Title font size
    "axes.labelsize": 10,                  # Label font size
    "legend.fontsize": 8,                  # Legend font size
    "xtick.labelsize": 8,                  # X-axis tick font size
    "ytick.labelsize": 8,                  # Y-axis tick font size
    # "figure.figsize": (3.5, 2.5),          # Default figure size for single-column plots
    # "figure.dpi": 300,                     # High resolution for publication
    # "figure.dpi": 150,                     # High resolution for publication
    "lines.linewidth": 1,                  # Line width
    "grid.linestyle": "--",                # Dashed grid lines
    "grid.color": "gray",                  # Grid line color
    "grid.alpha": 0.7,                     # Grid line transparency
    "axes.grid": True,                     # Enable grid by default
    "legend.loc": "best",                  # Default legend location
})

# Configure the font rendering with LaTeX for compatibility
rc('text.latex', preamble=r'\usepackage{amsmath}')  # Allows using AMS math symbols

# Helper function to plot in 3D the eigenvalues for a grid of theta_x, theta_y
def plotEigenvalueMeshHelper(grid,numTheta,N):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = [tuple(random.random() for _ in range(3)) for _ in range(N)]  # Random RGB colors

    # Plot a surface for each state
    for idx,color in enumerate(colors): 
        X = np.array([[grid[i, j][0] for j in range(numTheta)] for i in range(numTheta)])
        Y = np.array([[grid[i, j][1] for j in range(numTheta)] for i in range(numTheta)])
        Z = np.array([[grid[i, j][2][idx] for j in range(numTheta)] for i in range(numTheta)])

        # Plot the surface
        ax.plot_surface(X, Y, Z, color=color, edgecolor='none', alpha=1.0)

    # Set labels
    ax.set_xlabel('Theta-X')
    ax.set_ylabel('Theta-Y')
    ax.set_zlabel('Energy Value')

    # Show the plot
    plt.tight_layout()  # Ensure everything fits in the figure
    # plt.savefig("example_plot.pdf")  # Save in high-quality PDF format for papers

    plt.show()

# Accepts "centered" V_mn coefficients and plots the corresponding real potential 
def plotRandomPotential(inverse_potential):
    # must shift coefficients to corner from centered-spectrum
    shifted = np.fft.ifftshift(inverse_potential)
    real = np.fft.ifft2(shifted).real
    # real = inverse_potential.real

    plt.figure(figsize=(8, 6))
    plt.imshow(real, extent=[0, real.shape[1], 0, real.shape[0]],
               origin='lower', cmap='inferno')
    plt.colorbar(label="Magnitude of Real-Space Potential")
    plt.title("Random Gaussian White Noise Potential in Real Space")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# another function to plot the potential but this time in 3D
def plotRandomPotential3D(inverse_potential, plot_type="surface"):
    """
    Plots the random Gaussian white noise potential in 3D.

    Parameters:
    - inverse_potential: 2D array, input frequency-space data.
    - plot_type: str, the type of 3D plot ('surface', 'wireframe', 'scatter').
    """
    # Shift coefficients and perform inverse FFT to get real-space potential
    shifted = np.fft.ifftshift(inverse_potential)
    real = np.fft.ifft2(shifted).real

    # Create grid coordinates for plotting
    x = np.arange(real.shape[1])
    y = np.arange(real.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = real

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Choose the 3D plotting style
    if plot_type == "surface":
        surf = ax.plot_surface(X, Y, Z, cmap="inferno", edgecolor="none", alpha=0.9)
        fig.colorbar(surf, ax=ax, label="Magnitude of Real-Space Potential")
    elif plot_type == "wireframe":
        ax.plot_wireframe(X, Y, Z, color="orange", alpha=0.8)
    elif plot_type == "scatter":
        ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=Z.flatten(), cmap="inferno", marker="o")
    else:
        raise ValueError("Invalid plot_type. Choose 'surface', 'wireframe', or 'scatter'.")

    # Labels and Title
    ax.set_title("Random Gaussian White Noise Potential in Real Space")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Magnitude")

    plt.tight_layout()
    plt.show()

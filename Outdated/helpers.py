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
    "figure.dpi": 150,                     # High resolution for publication
    "lines.linewidth": 1,                  # Line width
    # "grid.linestyle": "--",                # Dashed grid lines
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

## TODO: make eigenvalueMeshHelper accept chern-numbers to colorcode the states!

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

def fix_eigenvector_phases(eigv):
    """
    Fixes the phase of each eigenvector so that the first component is real, to ensure smoothness
    of phase as we vary theta.
    
    Parameters:
        eigv (ndarray): Eigenvector matrix of shape (num_states, num_states),
                        where each column is an eigenvector.

    Returns:
        eigv_fixed (ndarray): Phase-corrected eigenvector matrix.
    """
    # Compute phase factors from the first row (first element of each eigenvector)
    phase_factors = np.exp(-1j * np.angle(eigv[0, :]))

    # Apply phase correction to each eigenvector (each column)
    eigv_fixed = eigv * phase_factors[np.newaxis, :]

    return eigv_fixed

def fix_eigenvector_phases(eigv):
    """
    Fixes the phase of each eigenvector so that the first component is real.
    """
    phase_factors = np.exp(-1j * np.angle(eigv[0, :]))
    return eigv * phase_factors[np.newaxis, :]

def plotSpecificEigenvalues(grid, numTheta, N, mismatch_indices=None):
    """
    Plots the 8 eigenvalue surfaces nearest to mismatches, highlighting mismatches in RED.
    
    - grid: Eigenvalue data grid [theta_x, theta_y, eigs] at each (i, j)
    - numTheta: Number of theta points along one axis
    - N: Total number of eigenvalues
    - mismatch_indices: List of mismatched eigenvalue indices to highlight (default: center region)
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate random colors for non-mismatched eigenvalues
    colors = [tuple(random.random() for _ in range(3)) for _ in range(N)]  
    high_contrast_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#17becf", 
    "#d62728", "#bcbd22", "#0000FF", "#008080", "#FF1493"]
    import colorcet as cc  
    colors = cc.glasbey_hv  # Highly distinct colors


    # colors = [plt.cm.Dark2(i % 8) for i in range(N)]  # Ensures distinct colors

    # Locate eigenvalues closest to the mismatches
    X = np.array([[grid[i, j][0] for j in range(numTheta)] for i in range(numTheta)])
    Y = np.array([[grid[i, j][1] for j in range(numTheta)] for i in range(numTheta)])
    
    # **Step 1: Identify indices to plot**
    if mismatch_indices is None or len(mismatch_indices) == 0:
        mismatch_indices = [N // 2]  # Default to center eigenvalue index

    # Find the range of eigenvalues to plot (center around mismatches)
    min_idx = max(0, min(mismatch_indices) - 4)
    max_idx = min(N, max(mismatch_indices) + 4)

    colorIDX=0
    for idx in range(min_idx, max_idx):  
        Z = np.array([[grid[i, j][2][idx] for j in range(numTheta)] for i in range(numTheta)])
        
        # Highlight mismatched eigenvalues in RED
        if idx in mismatch_indices:
            ax.plot_surface(X, Y, Z, color='red', edgecolor='none', alpha=0.9,linewidth=0,antialiased=True)
        else:
            ax.plot_surface(X, Y, Z, color=colors[colorIDX % len(colors)], edgecolor='none', alpha=0.9,linewidth=0,antialiased=True)
            colorIDX+=1

    # Set axis labels
    ax.set_xlabel(r'$\boldsymbol{\theta}_x$', fontsize=14, labelpad=12)
    ax.set_ylabel(r'$\boldsymbol{\theta}_y$', fontsize=14, labelpad=12)
    ax.set_zlabel(r'\textbf{Energy Value}', fontsize=14, labelpad=12)
    ax.set_title(r'\textbf{Eigenvalue Surfaces Near Mismatch}', fontsize=14)

    # ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.zaxis.label.set_rotation(90)  # Make Z-axis label readable


    plt.tight_layout()
    plt.show()
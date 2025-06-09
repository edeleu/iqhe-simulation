
import scipy.stats as stats
import os
import numpy as np
import csv
import timeit
from numba import njit, prange, jit
from timeit import default_timer as timer
from datetime import datetime
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
    "figure.figsize": (8, 5),          # Default figure size for single-column plots
    # "figure.dpi": 300,                     # High resolution for publication
    "figure.dpi": 150,                     # High resolution for publication
    "lines.linewidth": 1,                  # Line width
    # "grid.linestyle": "--",                # Dashed grid lines
    "grid.color": "gray",                  # Grid line color
    "grid.alpha": 0.6,                     # Grid line transparency
    "axes.grid": True,                     # Enable grid by default
    "legend.loc": "best",                  # Default legend location
})

# Configure the font rendering with LaTeX for compatibility
rc('text.latex', preamble=r'\usepackage{amsmath}')  # Allows using AMS math symbols
# plt.style.use("seaborn-darkgrid")

def load_trial_data(file_path):
    """Load trial data efficiently from a .npz file."""
    data = np.load(file_path)
    return {
        "PotentialMatrix": data["PotentialMatrix"],
        "ChernNumbers": data["ChernNumbers"],
        "SumChernNumbers": data["SumChernNumbers"],
        "eigs00": data["eigs00"],
        "eigs0pi": data["eigs0pi"],
        "eigsPi0": data["eigsPi0"],
        "eigsPipi": data["eigsPipi"]
    }

folderPath = "/Users/eddiedeleu/Downloads/N=1024_Mem"
filtered_eigsPipi = []
nonzeroEigs=[]

allEigSpacing = []
nonzeroEigSpacing = []
zeroEigSpacing = []
oneEigSpacing = []

for file_name in sorted(os.listdir(folderPath)):
    if file_name.endswith(".npz"):
        # print(file_name)
        file_path = os.path.join(folderPath, file_name)
        datas = load_trial_data(file_path)
        if np.isclose(datas["SumChernNumbers"], 1, atol=1e-5):
            currentEigs = datas["eigsPipi"]
            filtered_eigsPipi.append(currentEigs)

            differences = np.diff(currentEigs)
            allEigSpacing.append(differences)
            # Filter eigenvalues where corresponding ChernNumbers match the target value
            mask = ~np.isclose(datas["ChernNumbers"], 0, atol=1e-5)
            nonZeroValues = currentEigs[mask] # Select matching eigenvalues, non-zero chern
            nonzeroEigs.append(nonZeroValues) 

            NonZeroDifferences = np.diff(nonZeroValues)
            nonzeroEigSpacing.append(NonZeroDifferences)

            zeroValues = currentEigs[~mask]
            zeroEigSpacing.append(np.diff(zeroValues))

            oneEigSpacing.append(np.diff(currentEigs[np.isclose(datas["ChernNumbers"], -1, atol=1e-5)]))

        else:
            print(f"Trial {file_name} is not close!")
            print(datas["SumChernNumbers"])

eigenvalues = np.concatenate(filtered_eigsPipi) 
nonzeroEigenvalues = np.concatenate(nonzeroEigs)
NONZEROeigenvalueSpacing = np.concatenate(nonzeroEigSpacing)
eigenvalueSpacing = np.concatenate(allEigSpacing)

ZEROeigenvalueSpacing = np.concatenate(zeroEigSpacing)
ONEeigenvalueSpacing = np.concatenate(oneEigSpacing)

print(f"Extracted shape: {eigenvalues.shape}")
print(f"Extracted shape: {nonzeroEigenvalues.shape}")
print(f"Extracted shape: {ONEeigenvalueSpacing.shape}")

# top_20_indices = np.argsort(eigenvalueSpacing)[-100:]
# Extract the top 20 values using the indices
# top_20_values = eigenvalueSpacing[top_20_indices]
# Print the top 20 values
# print(top_20_values)

# pctile=np.percentile(eigenvalueSpacing,99.9)
# print(pctile)
# excludeExtremes = eigenvalueSpacing[(eigenvalueSpacing<=pctile)]

# plt.hist(excludeExtremes,density=True,bins=1000)
# plt.xlabel(r"Seperation of Eigenvalues, $s$", fontsize=14)
# plt.ylabel(r"Density of Seperations", fontsize=14)
# plt.title("Probability Density of Eigenvalue Seperations", fontsize=16)
# plt.show()

# METHOD of displaying fixed range
# num_bins = 1000
# bin_edges = np.linspace(0, 0.1, num_bins + 1)  # 1000 bins means 1001 edges
# bin_width = bin_edges[1] - bin_edges[0]

# # Compute histogram with proper density scaling
# counts, _ = np.histogram(NONZEROeigenvalueSpacing, bins=bin_edges)

# # Convert to density manually to avoid exclusion bias
# density = counts / (len(NONZEROeigenvalueSpacing) * bin_width)

# # Plot histogram with proper density normalization
# plt.bar(bin_edges[:-1], density, width=bin_width, alpha=0.8, label="Eigenvalues in Range")

# num_bins = 1000
# bin_edges = np.linspace(0, 0.1, num_bins + 1)  # 1000 bins means 1001 edges
# bin_width = bin_edges[1] - bin_edges[0]

# # Compute histogram with proper density scaling
# counts, _ = np.histogram(ONEeigenvalueSpacing, bins=bin_edges)

# # Convert to density manually to avoid exclusion bias
# density = counts / (len(ONEeigenvalueSpacing) * bin_width)

# # Plot histogram with proper density normalization
# plt.bar(bin_edges[:-1], density, width=bin_width, alpha=0.8, label="Eigenvalues in Range")

# plt.show()

def plotEigenvalueSpacing(eigenvalueSpacing,pctile=99):
    # Step 1: Compute central 95% range
    upper_bound = np.percentile(eigenvalueSpacing, pctile)  # Excludes extreme top % tails

    # Step 2: Define fixed bin edges within this range (1000 bins)
    num_bins = 1000
    bin_edges = np.linspace(0, upper_bound, num_bins + 1)  # 1001 edges

    # Step 3: Compute histogram counts over full dataset
    counts, _ = np.histogram(eigenvalueSpacing, bins=bin_edges)

    # Step 4: Convert to density over full dataset (ensures correct normalization)
    bin_width = bin_edges[1] - bin_edges[0]
    full_density = counts / (len(eigenvalueSpacing) * (bin_width))

    # Step 6: Plot histogram with renormalized density
    plt.bar(bin_edges[:-1], full_density, width=bin_width, label=f"Lower {pctile}%")

    # Aesthetics
    plt.xlabel(r"Separation of Eigenvalues, $s$", fontsize=14)
    plt.ylabel(r"Density of Separations", fontsize=14)
    plt.title("Probability Density of Eigenvalue Separations", fontsize=16)
    plt.legend(fontsize=10)
    plt.show()

    # Step 7: Confirm area in view sums to 1
    integrated_density = np.sum(full_density * bin_width)
    print(f"Integrated density over displayed range: {integrated_density:.5f} (should be ~1)")

def plotDOS(eigenvalues, nonzeroEigenvalues):
    # 1) Define common bin edges (e.g. 200 bins between min and max).
    num_bins = 300
    bin_edges = np.linspace(eigenvalues.min(), eigenvalues.max(), num_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]

    # 2) Plot the "all eigenvalues" histogram as a PDF (area = 1).
    counts_full, edges_full, _ = plt.hist(
        eigenvalues,
        bins=bin_edges,
        density=True,           # Make this a PDF
        alpha=0.8,
        label="All Eigenvalues"
    )

    # 3) Plot the subset histogram so that its total area = (#subset / #total).
    #    We do NOT use density=True; instead we supply weights:
    counts_subset, edges_subset, _ = plt.hist(
        nonzeroEigenvalues,
        bins=bin_edges,
        # Each value contributes 1/(N_total*bin_width)
        # so that the total area ends up = (#subset / #total).
        weights=np.ones_like(nonzeroEigenvalues) / (len(eigenvalues)*bin_width),
        alpha=0.6,
        label=r"$\ne0$ Chern Number"
    )

    # 4) Optionally, overlay a KDE for the full data (also integrates to 1).
    x = np.linspace(eigenvalues.min(), eigenvalues.max(), 200)
    kde = stats.gaussian_kde(eigenvalues, bw_method=0.2)
    plt.plot(x, kde(x), color='red', label="KDE Fit")

    # Aesthetics
    plt.xlabel(r"Energy Eigenvalues, $E$", fontsize=14)
    plt.ylabel(r"Density of States, $\rho(E)$", fontsize=14)
    plt.title("Density of States Histogram", fontsize=16)
    plt.legend()
    plt.show()

    # 5) Confirm the areas numerically
    area_full = np.sum(counts_full * np.diff(edges_full))
    area_subset = np.sum(counts_subset * np.diff(edges_subset))
    ratio = area_subset / area_full

    print(f"Area under the full histogram: {area_full:.6f} (should be ~1.0)")
    print(f"Area under the subset histogram: {area_subset:.6f}")
    print(f"Subset fraction of total: {ratio:.2%} (should be ~6-7%)")

# eigenvalueSpacing=ONEeigenvalueSpacing
# pctile=np.percentile(eigenvalueSpacing,99.9)
# print(pctile)

# plotEigenvalueSpacing(eigenvalueSpacing,90)
# plotEigenvalueSpacing(eigenvalueSpacing,95)
# plotEigenvalueSpacing(eigenvalueSpacing,99)
# plotEigenvalueSpacing(eigenvalueSpacing,99.9)
# plotEigenvalueSpacing(eigenvalueSpacing,99.99)

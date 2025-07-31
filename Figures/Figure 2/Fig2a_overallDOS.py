import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from tqdm import tqdm
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from scipy.stats import linregress

# (Keep the plot settings as before)
# Configure plot settings
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
    "axes.grid": True
})

def load_eigenvalues(folder_path):
    """Load eigenvalues from all .npz files in directory, filtering for SumChernNumbers=1"""
    valid_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    all_eigs = []
    nonzero_chern_eigs = []
    
    for fname in tqdm(valid_files, desc=f"Loading {os.path.basename(folder_path)}"):
        data = np.load(os.path.join(folder_path, fname))
        
        # Check if SumChernNumbers is 1
        if not np.isclose(data['SumChernNumbers'], 1, atol=1e-5):
            continue
        
        eigs = data['eigsPipi']
        chern_numbers = data['ChernNumbers']
        
        all_eigs.extend(eigs)
        nonzero_chern_eigs.extend(eigs[chern_numbers != 0])
    
    return np.array(all_eigs), np.array(nonzero_chern_eigs)

def plot_dos_comparison_final(system_sizes, base_path, output_file="Fig2a.pdf"):
    fig, ax1 = plt.subplots(1, 1, figsize=(3.4, 3))
    ratios = {}
    cmap = plt.cm.turbo
    color_norm = mcolors.Normalize(vmin=0, vmax=len(system_sizes)-1)
    colors = ["#332288", "#117733", "#88CCEE",
    # "#4575b4",  # indigo
    # "#1fa187",  # teal
    # "#65b300",  # olive green
    "#E69F00",  # orange
    "#d73027",  # vermillion
    "#740606"]  # black]


    for idx, n in enumerate(system_sizes):
        if n==1024 or n==2048:
            folder_path = os.path.join(base_path, f"N={n}_Mem")
        else:
            folder_path = os.path.join(base_path, f"N={n}")

        if not os.path.exists(folder_path):
            print(f"Directory {folder_path} not found, skipping")
            continue
            
        # Load data
        all_eigs, nonzero_chern_eigs = load_eigenvalues(folder_path)
        
        #Symmetrify data
        all_eigs = np.concatenate((all_eigs, -all_eigs))
        nonzero_chern_eigs = np.concatenate((nonzero_chern_eigs, -nonzero_chern_eigs))

        if len(all_eigs) == 0 or len(nonzero_chern_eigs) == 0:
            print(f"No valid data found for N={n}, skipping")
            continue
            
        # color = cmap(color_norm(idx))
        # Plot KDEs
        label = f"N={n}"
        color=colors[idx]

        x = np.linspace(all_eigs.min(), all_eigs.max(), 1000)
        kde_full = stats.gaussian_kde(all_eigs,bw_method=0.1)
        ax1.plot(x, kde_full(x), color=color, label=label, alpha=0.8)

        # Compute KDE for nonzero eigenvalues
        kde_subset = stats.gaussian_kde(nonzero_chern_eigs,bw_method=0.1)
        ax1.plot(x, kde_subset(x) * (len(nonzero_chern_eigs) / len(all_eigs)), linestyle='--', color=color, alpha=0.8)

        # Calculate ratio of maxima

        max_full = kde_full(x).max()
        max_subset = kde_subset(x).max()* (len(nonzero_chern_eigs) / len(all_eigs))
        # max_full = counts_full.max()
        # max_subset = counts_subset.max()
        print("Subset", max_subset)
        print("Full", max_full)
                
    # Format full DOS plot
    # ax1.set_title("Full Eigenvalue Spectrum")
    ax1.set_xlabel(r"$E$")
    ax1.set_ylabel(r"Normalized DOS $\rho(E)$")
    ax1.legend(loc='upper right', frameon=False)
        
    # Final adjustments
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


system_sizes = [64,128, 256, 512, 1024, 2048]
base_path = "/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/"
plot_dos_comparison_final(system_sizes, base_path)

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from tqdm import tqdm

# Plot settings (same as before)
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
    # "grid.alpha": 0.3,
    # "axes.grid": True
})

def load_eigenvalues(folder_path):
    """
    Load eigenvalues from all .npz files in a folder.
    Only files with SumChernNumbers == 1 are accepted.
    Returns:
      all_eigs: all eigenvalues from accepted files.
      nonzero_chern_eigs: eigenvalues with nonzero ChernNumbers.
    """
    valid_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    all_eigs = []
    nonzero_chern_eigs = []
    
    for fname in tqdm(valid_files, desc=f"Loading {os.path.basename(folder_path)}"):
        data = np.load(os.path.join(folder_path, fname))
        if not np.isclose(data['SumChernNumbers'], 1, atol=1e-5):
            continue
        eigs = data['eigsPipi']
        chern_numbers = data['ChernNumbers']
        all_eigs.extend(eigs)
        nonzero_chern_eigs.extend(eigs[chern_numbers != 0])
    
    return np.array(all_eigs), np.array(nonzero_chern_eigs)

def compute_fwhm(x, y):
    """
    Compute the Full Width at Half Maximum (FWHM) for a given curve.
    Assumes that x is sorted and y is the corresponding KDE values.
    """
    max_y = y.max()
    half_y = max_y / 2.0
    # Find indices where the curve is above half max
    indices = np.where(y >= half_y)[0]
    if len(indices) == 0:
        return np.nan
    left_idx = indices[0]
    right_idx = indices[-1]
    fwhm = x[right_idx] - x[left_idx]
    return fwhm, x[left_idx], x[right_idx], half_y

def plot_dos_pdf(system_sizes, base_path, n_bins=100, output_pdf="DOS_per_system_size.pdf"):
 
    # Special folder naming for larger system sizes
    if system_sizes == 1024 or system_sizes == 2048:
        folder_path = os.path.join(base_path, f"N={system_sizes}_Mem")
    else:
        folder_path = os.path.join(base_path, f"N={system_sizes}")
    
    # Load data
    all_eigs, nonzero_chern_eigs = load_eigenvalues(folder_path)
        
    # Symmetrize the data (include negative eigenvalues)
    all_eigs = np.concatenate((all_eigs, -all_eigs))
    nonzero_chern_eigs = np.concatenate((nonzero_chern_eigs, -nonzero_chern_eigs))
    
    # Create figure with two subplots: left for full, right for nonzero
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 3))      
    
    # Define a common x-axis range for full spectrum histogram
    x_min_full = all_eigs.min()
    x_max_full = all_eigs.max()
    x_full = np.linspace(x_min_full, x_max_full, 1000)
    
    # Compute histogram for full eigenvalues
    ax.hist(all_eigs, bins=n_bins, density=True, alpha=0.5, color="#d73027",histtype='stepfilled')
    ax.hist(all_eigs, bins=n_bins, density=True, color="k", histtype="step")
    
    density, bin_edges = np.histogram(all_eigs, bins=n_bins, density=True)
    widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + widths / 2
    # plt.plot(
    #     bin_centers, density,
    #     marker='o',
    #     linestyle='none',          # No lines connecting points
    #     markersize=4.5,
    #     markerfacecolor="#d73027",  # Filled APS blue color
    #     markeredgewidth=.6,
    #     markeredgecolor="#711a15",        # Black outline
    #     # label=r"Mean $N_{C\ne0}$"
    # )


    # Compute KDE for full eigenvalues
    kde_full = stats.gaussian_kde(all_eigs,bw_method=0.1) # bw_method=0.2
    y_full = kde_full(x_full)
    ax.plot(x_full, y_full, color="#d73027", lw=1, label="KDE Fit",alpha=0.8)
    
    # Compute FWHM for full spectrum
    fwhm_full, left_full, right_full, half_full = compute_fwhm(x_full, y_full)
    # --- full spectrum FWHM ---
    ax.annotate(
        '', xy=(right_full, half_full), xytext=(left_full, half_full),
        arrowprops= dict(shrinkA=0, shrinkB=0, arrowstyle='<->,head_length=0.3,head_width=0.15', lw=0.8, color='black', mutation_scale=6))
    # ax.text((left_full + right_full)/2, half_full*1.05,
            # fr"$\Delta E_\mathrm{{FWHM}}={fwhm_full:.3f}$\n[{left_full:.2f},{right_full:.2f}]",
            # ha='center', va='bottom', fontsize=8)

    # ax.text(right_full + 0.1, half_full, rf"$\Delta E_{{\mathrm{{FWHM}}}} = {fwhm_full:.2f}$", ha='left', va='center', fontsize=8)
    # ax.text(right_full + 0.15+0.01, half_full-0.02, rf"$[{left_full:.2f},\,{right_full:.2f}]$", ha='left', va='center', fontsize=8)

    # Define a common x-axis range for nonzero chern eigenvalues
    x_min_nz = nonzero_chern_eigs.min()
    x_max_nz = nonzero_chern_eigs.max()
    x_nz = np.linspace(x_min_nz, x_max_nz, 1000)
    
    # Compute histogram for nonzero chern eigenvalues
    counts_nz, bin_edges = np.histogram(nonzero_chern_eigs, bins=n_bins, density=True)
    bin_width = bin_edges[1] - bin_edges[0]
    ax.hist(nonzero_chern_eigs, bins=n_bins, weights=np.ones_like(nonzero_chern_eigs) / (len(all_eigs)*bin_width), alpha=0.3,color="#d73027",histtype='stepfilled')
    
    # Compute KDE for nonzero chern eigenvalues
    kde_nz = stats.gaussian_kde(nonzero_chern_eigs,bw_method=0.1) # bw_method=0.2
    y_nz = kde_nz(x_nz) * (len(nonzero_chern_eigs) / len(all_eigs))
    ax.plot(x_nz, y_nz, color="#d73027", linestyle="dashed", lw=1, label="KDE Fit",alpha=0.8)
    
    # Compute FWHM for nonzero spectrum
    fwhm_nz,   left_nz,   right_nz,   half_nz   = compute_fwhm(x_nz,   y_nz)

    # --- C≠0 spectrum FWHM 
    ax.annotate(
    '', xy=(right_nz, half_nz), xytext=(left_nz, half_nz),
    arrowprops=dict(arrowstyle='<->,head_length=0.3,head_width=0.15', lw=0.8, color='gray', shrinkA=0, shrinkB=0,mutation_scale=5)) # dict(arrowstyle='<->', lw=0.8, color='gray', shrinkA=0, shrinkB=0, mutation_scale=5)

    # ax.text(right_nz + 0.15, half_nz,
    # rf"$\Delta E_{{\mathrm{{FWHM}}}}^{{(C\neq 0)}} = {fwhm_nz:.3f}$",
    # ha='left', va='center', fontsize=8, color='gray')
    # ax.text(right_nz + 0.15, half_nz-0.02,
    # rf"$[{left_nz:.2f},\,{right_nz:.2f}]$",
    # ha='left', va='center', fontsize=8, color='gray')
    
    y_arrow = y_full.max()*1.04          # slightly above the KDE
    # ax.annotate(
    #     '', xy=(0.03,  y_arrow), xytext=(-0.03, y_arrow),
    #     arrowprops=dict(arrowstyle='-', lw=0.5, color='blue',
    #                     shrinkA=0, shrinkB=0, mutation_scale=6)
    # )


    ax.vlines([-0.03, 0.03], ymin=0, ymax=[y_arrow, y_arrow], 
            colors='blue', linestyles='dotted',lw=0.5)
    # ax.text(0.55, y_arrow, r"$|E|<0.03$", ha='left', va='bottom',
    #     fontsize=8, color='blue')


    ax.text(0.05, 0.865, r"$N_\phi = 1024$",
        transform=ax.transAxes, fontsize=12, fontweight='bold',
        ha='left', va='top')


    # --- mini “legend” in upper-right ------------------------------------------
    x0, y0 = 0.98, 0.96           # anchor (axes-fraction coords)

    ax.text(x0, y0,
            r"$|E|\le 0.03$",        # line 1 – blue
            transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color='blue')

    ax.text(x0, y0-0.07,
            rf"$\Delta E_{{\mathrm{{FWHM}}}} = {fwhm_full:.3f}$",    # line 2 – black
            transform=ax.transAxes,
            ha='right', va='top', fontsize=8)

    ax.text(x0, y0-0.14,
            rf"$\Delta E_{{\mathrm{{FWHM}}}}^{{(C\neq 0)}} = {fwhm_nz:.3f}$",  # line 3 – gray
            transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color='gray')
    # ---------------------------------------------------------------------------

    ax.set_xlabel(r"$E$")
    ax.set_ylabel(r"Normalized DOS $\rho(E)$")
    #ax.legend()
    ax.set_xlim(-5.5,5.5)
    ax.set_ylim(0,0.265)

    fig.tight_layout()
    plt.savefig(output_pdf)

def main():
    # Configuration
    base_path = "/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/"
    system_sizes = 1024
    n_bins = 100  # Set your desired number of bins here
    
    # Generate PDF with one page per system size
    plot_dos_pdf(system_sizes, base_path, n_bins=n_bins, output_pdf="Figure2b.pdf")

if __name__ == "__main__":
    main()

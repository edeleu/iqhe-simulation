import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

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
    "grid.alpha": 0.3,
    "axes.grid": True
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
    return fwhm

def plot_dos_pdf(system_sizes, base_path, n_bins=50, output_pdf="DOS_per_system_size.pdf"):
    """
    For each system-size, load the eigenvalue data, and produce a PDF page
    with two subplots:
       1. Histogram and KDE-fit for the full eigenvalue spectrum.
       2. Histogram and KDE-fit for the nonzero Chern eigenvalue spectrum.
    Also, the FWHM of the KDE-fit and the total number of datapoints are annotated.
    """
    pp = PdfPages(output_pdf)
    
    for n in system_sizes:
        # Special folder naming for larger system sizes
        if n == 1024 or n == 2048:
            folder_path = os.path.join(base_path, f"N={n}_Mem")
        else:
            folder_path = os.path.join(base_path, f"N={n}")
            
        if not os.path.exists(folder_path):
            print(f"Directory {folder_path} not found, skipping N={n}")
            continue
        
        # Load data
        all_eigs, nonzero_chern_eigs = load_eigenvalues(folder_path)
        
        # Skip if no valid data
        if len(all_eigs) == 0 or len(nonzero_chern_eigs) == 0:
            print(f"No valid data found for N={n}, skipping")
            continue
        
        # Symmetrize the data (include negative eigenvalues)
        all_eigs = np.concatenate((all_eigs, -all_eigs))
        nonzero_chern_eigs = np.concatenate((nonzero_chern_eigs, -nonzero_chern_eigs))
        
        # Create figure with two subplots: left for full, right for nonzero
        fig, (ax_full, ax_nonzero) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(fr"DOS for System Size $N={n}$", fontsize=14)
        
        # Define a common x-axis range for full spectrum histogram
        x_min_full = all_eigs.min()
        x_max_full = all_eigs.max()
        x_full = np.linspace(x_min_full, x_max_full, 500)
        
        # Compute histogram for full eigenvalues
        counts_full, bin_edges = np.histogram(all_eigs, bins=n_bins, density=True)
        ax_full.hist(all_eigs, bins=n_bins, density=True, alpha=0.5, color='C0', label=f"Histogram ({len(all_eigs)} pts)")
        
        # Compute KDE for full eigenvalues
        kde_full = stats.gaussian_kde(all_eigs,bw_method=0.1) # bw_method=0.2
        y_full = kde_full(x_full)
        ax_full.plot(x_full, y_full, 'k-', lw=1, label="KDE Fit",alpha=0.8)
        
        # Compute FWHM for full spectrum
        fwhm_full = compute_fwhm(x_full, y_full)
        ax_full.axvline(x_full[np.argmax(y_full)], color='k', linestyle='--', lw=1)
        ax_full.text(0.05, 0.95, fr"FWHM = {fwhm_full:.3f}", transform=ax_full.transAxes, 
                     verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        ax_full.vlines([-0.03, 0.03], ymin=0, ymax=[0.26, 0.26], 
               colors='blue', linestyles='dotted', label="Central Window")

        ax_full.set_title("Full Eigenvalue Spectrum")
        ax_full.set_xlabel(r"Energy $E$")
        ax_full.set_ylabel(r"Density of States $\rho(E)$")
        ax_full.legend()
        
        # Define a common x-axis range for nonzero chern eigenvalues
        x_min_nz = nonzero_chern_eigs.min()
        x_max_nz = nonzero_chern_eigs.max()
        x_nz = np.linspace(x_min_nz, x_max_nz, 1000)
        
        # Compute histogram for nonzero chern eigenvalues
        counts_nz, bin_edges = np.histogram(nonzero_chern_eigs, bins=n_bins, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        ax_nonzero.hist(nonzero_chern_eigs, bins=n_bins, weights=np.ones_like(nonzero_chern_eigs) / (len(all_eigs)*bin_width), alpha=0.5, color='C1', label=f"Histogram ({len(nonzero_chern_eigs)} pts)")
        
        # Compute KDE for nonzero chern eigenvalues
        kde_nz = stats.gaussian_kde(nonzero_chern_eigs,bw_method=0.1) # bw_method=0.2
        y_nz = kde_nz(x_nz) * (len(nonzero_chern_eigs) / len(all_eigs))
        ax_nonzero.plot(x_nz, y_nz, 'k-', lw=1, label="KDE Fit",alpha=0.8)
        
        # Compute FWHM for nonzero spectrum
        fwhm_nz = compute_fwhm(x_nz, y_nz)
        ax_nonzero.axvline(x_nz[np.argmax(y_nz)], color='k', linestyle='--', lw=1)
        ax_nonzero.text(0.05, 0.95, fr"FWHM = {fwhm_nz:.3f}", transform=ax_nonzero.transAxes, 
                        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        
        ax_nonzero.vlines([-0.03, 0.03], ymin=0, ymax=[0.1, 0.1], 
               colors='blue', linestyles='dotted', label="Central Window")
        
        ax_nonzero.set_title("Nonzero Chern Eigenvalue Spectrum")
        ax_nonzero.set_xlabel(r"Energy $E$")
        ax_nonzero.set_ylabel(r"Density of States $\rho(E)$")
        ax_nonzero.legend()
        
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pp.savefig(fig)
        plt.close(fig)
        
    pp.close()
    print(f"PDF saved as {output_pdf}")

def main():
    # Configuration
    base_path = "/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/"
    system_sizes = [8, 16, 32, 64, 96, 128, 192, 256, 512, 1024, 2048]
    system_sizes = [1024]
    n_bins = 100  # Set your desired number of bins here
    
    # Generate PDF with one page per system size
    plot_dos_pdf(system_sizes, base_path, n_bins=n_bins, output_pdf="DOS_per_system_size_0.2bw_125bins.pdf")

if __name__ == "__main__":
    main()

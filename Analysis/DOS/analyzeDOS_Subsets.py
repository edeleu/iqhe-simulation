import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from tqdm import tqdm
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from scipy.stats import linregress
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize_scalar

# (Keep the plot settings as before)
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

def split_list(lst, M):
    """Split a list into M approximately equal subsets."""
    k, m = divmod(len(lst), M)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(M)]

def load_eigenvalues_from_files(folder_path, file_list):
    """
    Load eigenvalues from a given list of .npz files in a folder,
    filtering for SumChernNumbers == 1.
    """
    all_eigs = []
    nonzero_chern_eigs = []
    for fname in tqdm(file_list, desc=f"Processing subset in {os.path.basename(folder_path)}", leave=False):
        data = np.load(os.path.join(folder_path, fname))
        # Only accept if SumChernNumbers is 1 (with tolerance)
        if not np.isclose(data['SumChernNumbers'], 1, atol=1e-5):
            continue
        eigs = data['eigsPipi']
        chern_numbers = data['ChernNumbers']
        all_eigs.extend(eigs)
        nonzero_chern_eigs.extend(eigs[chern_numbers != 0])
    return np.array(all_eigs), np.array(nonzero_chern_eigs)

def plot_dos_comparison_subsets_pdf(system_sizes, base_path, n_subsets=5, output_pdf="dos_comparison_subsets.pdf"):
    """
    For each system size, split the data files into n_subsets.
    For each subset, compute the KDE for both the full eigenvalue spectrum
    and the nonzero Chern eigenvalue spectrum. Then, create a PDF page (one per system size)
    with two side-by-side subplots: the left for all eigenvalues and the right for nonzero.
    Each subset's KDE is overlaid and labeled with the subset number and number of datapoints.
    Returns a dictionary with keys as system size and values as (mean_ratio, std_error, list_of_ratios).
    """
    ratios_summary = {}  # {N: (mean_ratio, std_error, [subset ratios])}
    pdf = PdfPages(output_pdf)
    cmap = plt.cm.turbo
    color_norm = mcolors.Normalize(vmin=0, vmax=n_subsets-1)
    
    for idx, n in enumerate(system_sizes):
        # Set folder path (special case for N=1024,2048)
        if n == 1024 or n == 2048:
            folder_path = os.path.join(base_path, f"N={n}_Mem")
        else:
            folder_path = os.path.join(base_path, f"N={n}")
        if not os.path.exists(folder_path):
            print(f"Directory {folder_path} not found, skipping N={n}")
            continue
        
        valid_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])
        if len(valid_files) == 0:
            print(f"No .npz files found in {folder_path}, skipping N={n}")
            continue
        
        # Split files into n_subsets (if fewer files than subsets, some subsets may have 1 file)
        subsets = split_list(valid_files, n_subsets)
        subset_ratios = []
        
        # Create a new figure for this system size with two subplots (side-by-side)
        fig, (ax_full, ax_nonzero) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(fr"System Size $N={n}$: {n_subsets} Subsets", fontsize=14)
        
        for subset_idx, subset_files in enumerate(subsets):
            # Load eigenvalues for this subset
            all_eigs, nonzero_chern_eigs = load_eigenvalues_from_files(folder_path, subset_files)
            # Skip subset if no valid data loaded
            if len(all_eigs) == 0 or len(nonzero_chern_eigs) == 0:
                print(f"No valid data in subset {subset_idx+1} for N={n}, skipping this subset")
                continue
            
            # Symmetrize data: include negative eigenvalues
            # all_eigs = np.concatenate((all_eigs, -all_eigs))
            # nonzero_chern_eigs = np.concatenate((nonzero_chern_eigs, -nonzero_chern_eigs))
            
            # Define a common x-axis range for both spectra
            x_min = min(all_eigs.min(), nonzero_chern_eigs.min())
            x_max = max(all_eigs.max(), nonzero_chern_eigs.max())
            x = np.linspace(x_min, x_max, 500)
            
            # Compute KDEs with fixed bandwidth
            kde_full = stats.gaussian_kde(all_eigs, bw_method=0.1)
            kde_nonzero = stats.gaussian_kde(nonzero_chern_eigs, bw_method=0.1)
            
            y_full = kde_full(x)
            # rough_x_mode_full = x[np.argmax(y_full)]  # Rough mode estimate

            # Normalize nonzero density by ratio of counts
            norm_factor = len(nonzero_chern_eigs) / len(all_eigs)
            y_nonzero = kde_nonzero(x) * norm_factor
            
            # Calculate ratio of the maximum values
            max_full = y_full.max()
            max_nonzero = y_nonzero.max()

            # next, use optimization to find the actual maxima!
            optimized_max_full = np.max(-minimize_scalar(lambda x: -kde_full(x), 
                            bounds=(-2,2),
                            method='bounded').fun)
            
            print("Normal Max Full", max_full, "vs. Optimized Max", optimized_max_full)
            optimized_max_nonzero = np.max(-minimize_scalar(lambda x: -kde_nonzero(x), 
                bounds=(-2,2),
                method='bounded').fun*norm_factor)
            print("Normal Max Nonzero", max_nonzero, "vs. Optimized Max", optimized_max_nonzero)

            # ratio = max_nonzero / max_full
            ratio = max(optimized_max_nonzero,max_nonzero) / max(optimized_max_full,max_full)
            # print(ratio)
            subset_ratios.append(ratio)
            
            label_str = f"Subset {subset_idx+1} ({len(all_eigs)} pts)"
            label_str_nonzero = f"Subset {subset_idx+1} ({len(nonzero_chern_eigs)} pts)"
 
            color = cmap(color_norm(subset_idx))
            
            # Plot on left subplot (full spectrum) and right subplot (nonzero spectrum)
            ax_full.plot(x, y_full, color=color, alpha=0.5, linestyle='-', label=label_str)
            ax_nonzero.plot(x, y_nonzero, color=color, alpha=0.5, linestyle='-', label=label_str_nonzero)
        
        # Set titles, labels, and legends for each subplot
        ax_full.set_title("Full Eigenvalue Spectrum")
        ax_full.set_xlabel(r"Energy $E$")
        ax_full.set_ylabel(r"Density of States $\rho(E)$")
        ax_full.legend(loc='upper right', frameon=False, ncol=1)
        
        ax_nonzero.set_title("Nonzero Chern Spectrum")
        ax_nonzero.set_xlabel(r"Energy $E$")
        ax_nonzero.set_ylabel(r"Density of States $\rho(E)$")
        ax_nonzero.legend(loc='upper right', frameon=False, ncol=1)
        
        # Compute mean and standard error for this system size (if any valid subsets)
        subset_ratios = np.array(subset_ratios)
        if len(subset_ratios) > 0:
            mean_ratio = subset_ratios.mean()
            std_error = subset_ratios.std(ddof=1) / np.sqrt(len(subset_ratios))
            ratios_summary[n] = (mean_ratio, std_error, subset_ratios)
            print(f"N={n}: Mean ratio = {mean_ratio:.4f} with standard error {std_error:.4f}")
        else:
            print(f"No valid subsets for N={n}.")
        
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)
    
    pdf.close()
    print(f"PDF saved as {output_pdf}")
    return ratios_summary

def plot_ratios_with_error(ratios_summary, output_file="ratios_plot_with_error.pdf"):
    """
    Plot the average ratio for each system size with error bars.
    Also perform a weighted linear fit (using standard errors) and plot the fitted line.
    """
    # Sort system sizes
    xs = sorted(ratios_summary.keys())
    # Build arrays for plotting: use inverse system size as x
    invN = np.array([1.0/float(n) for n in xs])
    y = np.array([ratios_summary[n][0] for n in xs])
    yerr = np.array([ratios_summary[n][1] for n in xs])
    
    # Perform a weighted linear fit: model Q = slope*(1/N) + intercept.
    def linear_model(x, slope, intercept):
        return slope * x + intercept
    
    popt, pcov = curve_fit(linear_model, invN, y, sigma=yerr, absolute_sigma=True)
    slope, intercept = popt
    slope_err, intercept_err = np.sqrt(np.diag(pcov))
    
    print(f"Weighted linear fit results:")
    print(f"  Slope = {slope:.4f} +/- {slope_err:.4f}")
    print(f"  Intercept = {intercept:.4f} +/- {intercept_err:.4f}")
    
    # Generate fitted curve
    x_fit = np.linspace(invN.min(), invN.max(), 500)
    y_fit = linear_model(x_fit, slope, intercept)
    
    # Plot with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(invN, y, yerr=yerr, fmt='o', label="Average Ratio (with SE)")
    plt.plot(x_fit, y_fit, 'r--', label=fr"Fit: $Q = {slope:.4f}\cdot(1/N)+{intercept:.4f}$")
    plt.xlabel(r"Inverse Number of States, $1/N_{\phi}$", fontsize=14)
    plt.ylabel("Ratio of DOS Maxima (Nonzero / Full)", fontsize=14)
    plt.title("Ratio of DOS Maxima vs Inverse Number of States", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    # Configuration
    base_path = "/Users/eddiedeleu/Desktop/FinalData"
    # system_sizes = [8, 16, 32, 64, 96, 128, 192, 256, 512, 1024, 2048]
    system_sizes = [64, 96, 128, 192, 256, 512, 1024, 2048]

    n_subsets = 10  # Change this value as needed
    
    # Generate DOS comparison plots with subsets (each system size on a separate PDF page)
    ratios_summary = plot_dos_comparison_subsets_pdf(system_sizes, base_path, n_subsets=n_subsets,
                                                     output_pdf="dos_comparison_subsets_0.1_currentBig.pdf")
    
    # Plot average ratios with error bars and perform weighted linear fit
    plot_ratios_with_error(ratios_summary, output_file="ratios_plot_with_error_0.1_current1Big.pdf")
    
    # Print summary of ratios for each system size
    print("\nSummary of Ratio (Nonzero / Full) per System Size:")
    for n in sorted(ratios_summary.keys()):
        mean_ratio, std_error, subset_ratios = ratios_summary[n]
        print(f"N={n}: Mean ratio = {mean_ratio:.4f} +/- {std_error:.4f} (from {len(subset_ratios)} subsets)")

if __name__ == "__main__":
    main()
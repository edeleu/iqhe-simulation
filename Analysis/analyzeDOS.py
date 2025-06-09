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

def plot_dos_comparison_final(system_sizes, base_path, output_file="dos_comparison_finalv2.pdf"):
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
    "#000000"]  # black]


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
    

def plot_dos_comparison(system_sizes, base_path, output_file="dos_comparisonSymmetricUpdatedNewAUTO.pdf"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    ratios = {}
    cmap = plt.cm.turbo
    color_norm = mcolors.Normalize(vmin=0, vmax=len(system_sizes)-1)
    
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
            
        # PLOTTING CODE!
        # num_bins = 100
        # bin_edges = np.linspace(all_eigs.min(), all_eigs.max(), num_bins + 1)
        # bin_width = bin_edges[1] - bin_edges[0]

        # # Plot histogram for all eigenvalues as a PDF
        # counts_full, edges_full, _ = ax1.hist(
        #     all_eigs,
        #     bins=bin_edges,
        #     density=True,           # Make this a PDF
        #     alpha=0.8,
        #     label="All Eigenvalues"
        # )

        # # Plot histogram for nonzero-eigenvalues subset
        # counts_subset, edges_subset, _ = ax2.hist(
        #     nonzero_chern_eigs,
        #     bins=bin_edges,
        #     weights=np.ones_like(nonzero_chern_eigs) / (len(all_eigs)*bin_width),
        #     alpha=0.6,
        #     label=r"$\ne0$ Chern Number"
        # )

        # Compute KDE for all eigenvalues
        color = cmap(color_norm(idx))
        # Plot KDEs
        label = f"N={n}"

        x = np.linspace(all_eigs.min(), all_eigs.max(), 500)
        kde_full = stats.gaussian_kde(all_eigs,bw_method=0.1)
        ax1.plot(x, kde_full(x), color=color, label=label, alpha=0.8)

        # Compute KDE for nonzero eigenvalues
        kde_subset = stats.gaussian_kde(nonzero_chern_eigs,bw_method=0.1)
        ax2.plot(x, kde_subset(x) * (len(nonzero_chern_eigs) / len(all_eigs)), color=color, label=label, alpha=0.8)

        # Calculate ratio of maxima

        max_full = kde_full(x).max()
        max_subset = kde_subset(x).max()* (len(nonzero_chern_eigs) / len(all_eigs))
        # max_full = counts_full.max()
        # max_subset = counts_subset.max()
        ratios[n] = max_subset / max_full
        print("Subset", max_subset)
        print("Full", max_full)

            # === Compute FWHM for Full Data ===
            # kde_y_full = kde_full(x)
            # half_max_full = np.max(kde_y_full) / 2
            # mask_full = kde_y_full >= half_max_full
            # fwhm_x_lower_full, fwhm_x_upper_full = x[mask_full][0], x[mask_full][-1]

            # # Plot FWHM for full data
            # plt.vlines([fwhm_x_lower_full, fwhm_x_upper_full], ymin=0, ymax=[half_max_full, half_max_full], colors='black', linestyles='dotted', label="FWHM (All Data)")

            # # === Compute FWHM for Subset Data ===
            # kde_y_subset = kde_subset(x) * (len(nonzeroEigenvalues) / len(eigenvalues))
            # half_max_subset = np.max(kde_y_subset) / 2
            # mask_subset = kde_y_subset >= half_max_subset
            # fwhm_x_lower_subset, fwhm_x_upper_subset = x[mask_subset][0], x[mask_subset][-1]

                
    # Format full DOS plot
    ax1.set_title("Full Eigenvalue Spectrum")
    ax1.set_xlabel(r"Energy $E$")
    ax1.set_ylabel(r"Density of States $\rho(E)$")
    ax1.legend(loc='upper right', frameon=False, ncol=2)
    
    # Format nonzero Chern DOS plot
    ax2.set_title("Nonzero Chern Number Spectrum")
    ax2.set_xlabel(r"Energy $E$")
    ax2.set_ylabel(r"Density of States $\rho(E)$")
    ax2.legend(loc='upper right', frameon=False, ncol=2)
    
    # Final adjustments
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return ratios

def plot_ratios(ratios, output_file="ratios_plotSymm_auto_updatednew1.pdf"):
    plt.figure(figsize=(10, 6))
    x = sorted(ratios.keys())
    y = [ratios[k] for k in x]
    invX = [1/z for z in x]

    result = linregress(invX, y)

    slope_uncertainty = result.stderr
    intercept_uncertainty = result.intercept_stderr

    slope = result.slope
    intercept = result.intercept
    r = result.rvalue
    print(f"Slope: {result.slope:.3f} +/- {slope_uncertainty:.3f}")
    print(f"Intercept: {result.intercept:.3f} +/- {intercept_uncertainty:.3f}")


    # Print fit quality metrics
    print(f"R-squared: {r:.4f}")
    invN = np.linspace(0, max(invX), 500)
    Q_fitted = slope*invN+intercept

    plt.plot([1/z for z in x], y, 'o', label="Ratio of DOS Maxima")
    plt.plot(invN, Q_fitted, 'r--', label=fr"Fit: $Q = {slope:.4f} / N_\phi+{{{intercept:.4f}}}$")

    plt.xlabel(r"Inverse Number of States, $1/N_{\phi}$", fontsize=14)
    plt.ylabel("Ratio of DOS Maxima (Non-Zero Chern / All Chern)")
    plt.title("Ratio of DOS Maxima vs Inverse Number of States")
    # plt.xscale('log')
    # plt.yscale('log')

    # plt.xticks(x, labels=[str(N) for N in x]) 
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file)
    # plt.show()
    plt.close()


def plot_ratiosOLD(ratios, output_file="ratios_plotVx.pdf"):
    plt.figure(figsize=(10, 6))
    x = sorted(ratios.keys())
    y = [ratios[k] for k in x]

    popt, pcov = curve_fit(power_law, x, y)

    # Extract A and b
    A_fit, b_fit = popt
    A_err, b_err = np.sqrt(np.diag(pcov))  # Standard deviations of A and b
    # Print results with uncertainties
    print(f"Fitted parameters: A = {A_fit:.4f} ± {A_err:.4f}, b = {b_fit:.4f} ± {b_err:.4f}")

    # Generate smooth curve for fitting
    N_fit = np.logspace(min(x), max(x), 500)
    Q_fitted = power_law(N_fit, A_fit, b_fit)

    # Compute fitted values
    Q_approx = power_law(x, A_fit, b_fit)

    # Compute R-squared
    SS_res = np.sum((y - Q_approx) ** 2)  # Residual sum of squares
    print(SS_res)
    SS_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
    R_squared = 1 - (SS_res / SS_tot)

    # Compute RMSE
    RMSE = np.sqrt(np.mean((y - Q_approx) ** 2))
    # Print fit quality metrics
    print(f"R-squared: {R_squared:.4f}")
    print(f"RMSE: {RMSE:.4f}")

    plt.plot(x, y, 'o', label="Ratio of DOS Maxima")
    plt.plot(N_fit, Q_fitted, 'r--', label=fr"Fit: $Q = {A_fit:.4f} N_\phi^{{{b_fit:.4f}}}$")

    plt.xlabel(r"Number of States, $N_{\phi}$", fontsize=14)
    plt.ylabel("Ratio of DOS Maxima (Non-Zero Chern / All Chern)")
    plt.title("Ratio of DOS Maxima vs Number of States")
    plt.xscale('log')
    plt.yscale('log')

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def power_law(N, A, b):
    return A * N**b

def linear_law(x, A, b):
    return A*x+b

def plot_ratios_Only():
    plt.figure(figsize=(10, 6))
    x = [8,16,32,64,96,128,192, 256, 512, 1024, 2048]
    invX = [1/z for z in x]
    # y = [2.7673,3.1174,3.4924,3.7449,3.9690,4.3356,4.5623,5.2235,6.0503,6.9758]
    y = [0.43835,0.4165,0.4024,0.3925,0.3898,0.3907,0.3934,0.3887,0.3869,0.3902,0.3912]
    # popt, pcov = curve_fit(linear_law, x, y)

    # # Extract A and b
    # A_fit, b_fit = popt
    # A_err, b_err = np.sqrt(np.diag(pcov))  # Standard deviations of A and b
    # # Print results with uncertainties
    # print(f"Fitted parameters: A = {A_fit:.4f} ± {A_err:.4f}, b = {b_fit:.4f} ± {b_err:.4f}")

    # # Generate smooth curve for fitting
    # N_fit = np.linspace(min(x), max(x), 500)
    # Q_fitted = linear_law(N_fit, A_fit, b_fit)

    # # Compute fitted values
    # Q_approx = linear_law(x, A_fit, b_fit)

    # # Compute R-squared
    # SS_res = np.sum((y - Q_approx) ** 2)  # Residual sum of squares
    # print(SS_res)
    # SS_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
    # R_squared = 1 - (SS_res / SS_tot)

    # # Compute RMSE
    # RMSE = np.sqrt(np.mean((y - Q_approx) ** 2))
    result = linregress(invX, y)

    slope_uncertainty = result.stderr
    intercept_uncertainty = result.intercept_stderr

    slope = result.slope
    intercept = result.intercept
    r = result.rvalue 
    print(f"Slope: {result.slope:.3f} +/- {slope_uncertainty:.3f}")
    print(f"Intercept: {result.intercept:.3f} +/- {intercept_uncertainty:.3f}")


    # Print fit quality metrics
    print(f"R-squared: {r:.4f}")
    # print(f"RMSE: {RMSE:.4f}")
    invN = np.linspace(0, max(invX), 500)
    Q_fitted = slope*invN+intercept

    plt.plot([1/z for z in x], y, 'o', label="Ratio of DOS Maxima")
    plt.plot(invN, Q_fitted, 'r--', label=fr"Fit: $Q = {slope:.4f} / N_\phi+{{{intercept:.4f}}}$")

    plt.xlabel(r"Inverse Number of States, $1/N_{\phi}$", fontsize=14)
    plt.ylabel("Ratio of DOS Maxima (Non-Zero Chern / All Chern)")
    plt.title("Ratio of DOS Maxima vs Inverse Number of States")
    # plt.xscale('log')
    # plt.yscale('log')

    # plt.xticks(x, labels=[str(N) for N in x]) 
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("FinalFinal.pdf")
    plt.show()

    # plt.close()

def main():
    # Configuration
    base_path = "/Users/eddiedeleu/Desktop/FinalData"
    system_sizes = [8,16,32,64,96,128,192, 256, 512, 1024, 2048]

    # Generate plots and compute ratios
    ratios = plot_dos_comparison(system_sizes, base_path)
    
    # Plot ratios
    plot_ratios(ratios)
    
    # Print results
    print("\nMaximum KDE Ratios (subset/full):")
    for n in sorted(ratios.keys()):
        print(f"N={n}: {ratios[n]:.4f}")

if __name__ == "__main__":
    main()
    # plot_ratios_Only()

    # system_sizes = [64,128, 256, 512, 1024, 2048]
    # base_path = "/Users/eddiedeleu/Desktop/FinalData"
    # plot_dos_comparison_final(system_sizes, base_path)

# N=150 bins
# N=8: 0.4427, 4399 with 100 
# N=16: 0.4149
# N=32: 0.4051
# N=64: 0.3954
# N=96: 0.3934
# N=128: 0.3945
# N=192: 0.3947
# N=256: 0.3906
# N=512: 0.3875
# N=1024: 0.3885
# N=2048: 0.3876
# Slope: 0.429 +/- 0.016
# Intercept: 0.389 +/- 0.001
# R-squared: 0.9939

## 0.2 Method!
# Slope: 0.534 +/- 0.024
# Intercept: 0.387 +/- 0.001
# R-squared: 0.9907

# Maximum KDE Ratios (subset/full):
# N=8: 0.4566
# N=16: 0.4165
# N=32: 0.4024
# N=64: 0.3925
# N=96: 0.3898
# N=128: 0.3907
# N=192: 0.3934
# N=256: 0.3887
# N=512: 0.3869
# N=1024: 0.3902
# N=2048: 0.3912
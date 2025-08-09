# Figure2_ABC_combined.py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from tqdm import tqdm
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from scipy.stats import linregress
from matplotlib import rc

# ── Plot settings (shared) ───────────────────────────────────────────────
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
})
rc("text.latex", preamble=r"\usepackage{amsmath}")

# ── Common helper to load eigenvalues ─────────────────────────────────────
def load_eigenvalues(folder_path):
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

# ── Fig 2a plotting function ─────────────────────────────────────────────
def compute_fwhm(x, y):
    max_y = y.max()
    half_y = max_y / 2.0
    indices = np.where(y >= half_y)[0]
    if len(indices) == 0:
        return np.nan, None, None, None
    left_idx = indices[0]
    right_idx = indices[-1]
    return x[right_idx] - x[left_idx], x[left_idx], x[right_idx], half_y

def plot_fig2a(ax, system_size, base_path, n_bins=100):
    folder_path = os.path.join(base_path, f"N={system_size}_Mem" if system_size in (1024, 2048) else f"N={system_size}")
    all_eigs, nonzero_chern_eigs = load_eigenvalues(folder_path)
    all_eigs = np.concatenate((all_eigs, -all_eigs))
    nonzero_chern_eigs = np.concatenate((nonzero_chern_eigs, -nonzero_chern_eigs))
    # Define a common x-axis range for full spectrum histogram
    x_min_full = all_eigs.min()
    x_max_full = all_eigs.max()
    x_full = np.linspace(x_min_full, x_max_full, 1200)
    
    # Compute histogram for full eigenvalues
    ax.hist(all_eigs, bins=n_bins, density=True, alpha=0.5, color="#d73027",histtype='stepfilled')    
    density, bin_edges = np.histogram(all_eigs, bins=n_bins, density=True)
    widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + widths / 2

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

    # Define a common x-axis range for nonzero chern eigenvalues
    x_min_nz = nonzero_chern_eigs.min()
    x_max_nz = nonzero_chern_eigs.max()
    x_nz = np.linspace(x_min_nz, x_max_nz, 600)
    
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

    y_arrow = y_full.max()*1.04          # slightly above the KDE
    ax.vlines([-0.03, 0.03], ymin=0, ymax=[y_arrow, y_arrow], 
            colors='blue', linestyles='dotted',lw=0.5)
    ax.text(0.05, 0.865, r"$N_\phi = 1024$",
        transform=ax.transAxes, fontsize=12, fontweight='bold', ha='left', va='top')

    # --- mini “legend” in upper-right ------------------------------------------
    x0, y0 = 0.98, 0.96           # anchor (axes-fraction coords)
    ax.text(x0, y0, r"$|E|\le 0.03$",        # line 1 – blue
            transform=ax.transAxes, ha='right', va='top', fontsize=8, color='blue')
    ax.text(x0, y0-0.07, rf"$\Delta E_{{\mathrm{{FWHM}}}} = {fwhm_full:.3f}$",    # line 2 – black
            transform=ax.transAxes, ha='right', va='top', fontsize=8)
    ax.text(x0, y0-0.14, rf"$\Delta E_{{\mathrm{{FWHM}}}}^{{(C\neq 0)}} = {fwhm_nz:.3f}$",  # line 3 – gray
            transform=ax.transAxes, ha='right', va='top', fontsize=8, color='gray')

    ax.set_xlabel(r"$E$")
    ax.set_ylabel(r"Normalized DOS $\rho(E)$")
    ax.set_xlim(-5.5,5.5)
    ax.set_ylim(0,0.265)    

# ── Fig 2b plotting function ─────────────────────────────────────────────
def plot_fig2b(ax, system_sizes, base_path):
    colors = ["#332288", "#117733", "#88CCEE", "#E69F00", "#d73027", "#D479D1"]  
    for idx, n in enumerate(system_sizes):
        if n in (1024, 2048):
            folder_path = os.path.join(base_path, f"N={n}_Mem")
        else:
            folder_path = os.path.join(base_path, f"N={n}")

        if not os.path.exists(folder_path):
            continue
            
        all_eigs, nonzero_chern_eigs = load_eigenvalues(folder_path)
        all_eigs = np.concatenate((all_eigs, -all_eigs))
        nonzero_chern_eigs = np.concatenate((nonzero_chern_eigs, -nonzero_chern_eigs))
        if len(all_eigs) == 0 or len(nonzero_chern_eigs) == 0:
            continue
            
        label = rf"$N_\phi={n}$"
        color = colors[idx]

        x = np.linspace(all_eigs.min(), all_eigs.max(), 1200)
        kde_full = stats.gaussian_kde(all_eigs, bw_method=0.1)
        ax.plot(x, kde_full(x), color=color, label=label, alpha=0.8, lw=0.75)

        x = np.linspace(nonzero_chern_eigs.min(), nonzero_chern_eigs.max(), 600)
        kde_subset = stats.gaussian_kde(nonzero_chern_eigs, bw_method=0.1)
        ax.plot(x, kde_subset(x) * (len(nonzero_chern_eigs) / len(all_eigs)),
                linestyle='--', color=color, alpha=0.8, lw=0.75)

    ax.set_xlabel(r"$E$")
    ax.set_ylabel(r"Normalized DOS $\rho(E)$")
    ax.legend(loc='upper right', frameon=False, handlelength=1.2)

    ax.annotate(r"$C\ne0$ states", xy=(0, 0.1), xycoords='data',
                xytext=(-40, 0), textcoords='offset points',
                color='black', fontsize=8, ha='right', va='center',
                arrowprops=dict(arrowstyle='->', color="#000000", lw=0.8))
    
    ax.annotate(r"All states", xy=(0, 0.2533), xycoords='data',
                xytext=(-35, 0), textcoords='offset points',
                color='black', fontsize=8, ha='right', va='top',
                arrowprops=dict(arrowstyle='->', color="#000000", lw=0.8))
    ax.set_ylim(-0.005, 0.265)

# ── Fig 2c plotting function ─────────────────────────────────────────────
def power_law(N, A, b):
    # return A * N**b
    return A * N**(1-1/(2*b))

def plot_fig2c(ax):
    dataMeans = [1.497231496,
    2.409899074,
    4.138923757,
    7.216011783,
    10.00586747,
    12.59989761,
    17.43272707,
    21.91971655,
    38.13765199,
    66.20554142,
    114.6200583]

    data = [0.000480347,
    0.0011296,
    0.002546886,
    0.004811596,
    0.007665335,
    0.009622746,
    0.015306903,
    0.019343819,
    0.035747716,
    0.07609329,
    0.1555410
    ]

    # Example data (replace with actual values from your calculations)
    num_states_values = np.array([8, 16, 32, 64,96, 128,192,256,512,1024,2048])  # N_phi values
    mean_nc_values = np.array(dataMeans)  # Mean N_c
    std_error = np.array(data)  # Standard Error of N_c

    # Fit power law using scipy's curve_fit
    # popt, pcov = curve_fit(power_law, num_states_values, mean_nc_values)
    popt, pcov = curve_fit(power_law, num_states_values[2:], mean_nc_values[2:],sigma=std_error[2:], absolute_sigma=True)

    # Extract A and b
    A_fit, b_fit = popt
    A_err, b_err = np.sqrt(np.diag(pcov))  # Standard deviations of A and b

    # Print results with uncertainties
    print(f"Fitted parameters: $A = {A_fit:.4f} ± {A_err:.4f}$, $b = {b_fit:.4f} ± {b_err:.4f}$")

    # Generate smooth curve for fitting
    N_fit = np.linspace(min(num_states_values), max(num_states_values), 100)
    NC_FIT = power_law(N_fit, A_fit, b_fit)

    # Compute fitted values
    NC_approx = power_law(num_states_values, A_fit, b_fit)

    # Compute R-squared
    SS_res = np.sum((mean_nc_values - NC_approx) ** 2)  # Residual sum of squares
    print(SS_res)
    SS_tot = np.sum((mean_nc_values - np.mean(mean_nc_values)) ** 2)  # Total sum of squares
    R_squared = 1 - (SS_res / SS_tot)

    # Compute RMSE
    RMSE = np.sqrt(np.mean((mean_nc_values - NC_approx) ** 2))

    # Print fit quality metrics
    print(f"R-squared: {R_squared:.4f}")
    print(f"RMSE: {RMSE:.4f}")

    ax.plot(
        num_states_values, mean_nc_values,
        marker='o',
        linestyle='none',          # No lines connecting points
        markersize=4.5,
        markerfacecolor="#A0A0A0",  # Filled APS blue color
        markeredgewidth=.6,
        markeredgecolor="#000000")
    ax.plot(N_fit, NC_FIT, 'b--', color="#4577f6", linewidth=1.2, label=(r"$N_{C\ne0} = A\cdot N_\phi ^{1-\frac{1}{2\nu}}$"))

    # Aesthetics
    ax.set_xlabel(r"$N_{\phi}$")
    ax.set_ylabel(r"$\langle N_{C\ne0} \rangle$") # Mean Number of Nonzero Chern States, 
    ax.set_xscale("log")  # Use log scale if necessary
    ax.set_yscale("log")  # Use log scale if necessary

    ax.set_xticks([8, 16, 32, 64, 128,256,512,1024,2048], labels=[str(N) for N in [8, 16, 32, 64, 128,256,512,1024,2048]])  # Label x-axis
    ax.set_yticks([1, 5, 10, 50, 100], labels=["1", "5", "10", "50", "100"])

    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.94), frameon=False)
    param_text = (
        rf"$A = {A_fit:.4f} \pm {A_err:.4f}$" + "\n" +
        rf"$\nu = {b_fit:.4f} \pm {b_err:.4f}$" + "\n" +
        rf"$R^2 = {R_squared:.4f}$"
    )
    ax.text(
        0.06, 0.72, param_text,
        transform=ax.transAxes,
        fontsize=9,
        linespacing=1.8,  
        verticalalignment='top')

# ── Combined figure layout ───────────────────────────────────────────────
def main():
    base_path = "/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/"
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3))
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.95, wspace=0.4)

    # plot_fig2a(axes[0], 1024, base_path, n_bins=100)
    plot_fig2b(axes[1], [64,128, 256, 512, 1024, 2048], base_path)
    plot_fig2c(axes[2])

    plt.savefig("Figure2_combined.pdf")
    plt.close(fig)

if __name__ == "__main__":
    main()

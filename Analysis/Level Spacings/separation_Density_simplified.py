#WIP to simplify code and get rid of fitter stuff. Not working

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from matplotlib import gridspec, rc
import fitter
import scipy.stats as st
from scipy import stats

from scipy.stats import rv_continuous
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# -- Custom Distributions for Wigner Surmise --
def overlay_gue_curve(ax, s_max=6, num_points=1000, label="GUE", color="green", linestyle="--"):
    s = np.linspace(0, s_max, num_points)
    # p_s = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
    # p_s = np.exp(-1.65*(s-0.6))
    p_s = s*np.exp(-s)

    ax.plot(s, p_s, label=label, color=color, linestyle=linestyle)

# Configure plot settings (as before)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "lines.linewidth": 1,
    "grid.alpha": 0.3,
    "axes.grid": True
})

rc('text.latex', preamble=r'\usepackage{amsmath}')

# Chern filters (as before)
CHERN_FILTERS = {
    r'All Chern': None,
    r'C = $0$ or $+1$': [0, 1],
    r'C = $0$ or $-1$': [0, -1],
    r'C = $0$': [0],
    r'C = $-1$': [-1],
    r'C = $+1$': [1],
    r'$|$C$|$ = 1': [-1, 1],
    r'C $\ne 0$': [-3,-2,-1, 1,2,3]
}

# process_trial, normalize_separations, get_dynamic_bin_count (as before)
def process_trial(data, energy_range, chern_filters):
    eigs = data['eigs']
    chern = data['chern']
    mask = (eigs >= energy_range[0]) & (eigs <= energy_range[1])
    trial_eigs = eigs[mask]
    trial_chern = chern[mask]
    results = {}
    for name, cf in chern_filters.items():
        if cf is None:
            filtered_eigs = trial_eigs
        else:
            c_mask = np.isin(trial_chern, cf)
            filtered_eigs = trial_eigs[c_mask]
        separations = np.diff(filtered_eigs) if len(filtered_eigs) > 1 else np.array([])
        results[name] = separations
    return results

def normalize_separations(separations):
    if len(separations) == 0:
        return separations, 0
    avg_separation = np.mean(separations)
    normalized_seps = separations / avg_separation
    return normalized_seps, avg_separation

# def get_dynamic_bin_count(n_points):
#     # return 100
#     if n_points >= 50000:
#         return 500
#     elif n_points >= 10000:
#         return int(100 + (n_points - 10000) * 400 / 40000)
#     elif n_points >= 5000:
#         return int(50 + (n_points - 5000) * 50 / 5000)
#     else:
#         return max(20, int(n_points / 100))

def get_dynamic_bin_count(n_points):
    """
    Returns a dynamic number of bins based on the number of data points.
    For n_points ~1,300, this gives ~30 bins.
    For n_points ~1,300,000, this gives ~250 bins (maximum).
    """
    A = 3.25
    alpha = 0.31
    bins = int(np.ceil(A * n_points**alpha))
    return min(bins, 250)


def generate_plots_save(energy_range, all_separations, pdf):
    # fig = plt.figure(figsize=(20, 5 * len(all_separations)))
    # fig.suptitle(f"Energy Range: [{energy_range[0]:.3f}, {energy_range[1]:.3f}]", y=0.98, fontsize=16)

    for row, (name, seps) in enumerate(all_separations.items()):
        if seps.size == 0:
            continue

        normalized_seps, avg_sep = normalize_separations(seps)
        n_bins = get_dynamic_bin_count(normalized_seps.size)

        best_distrs = []
        best_fit_stats = {}
        best_fit_params = {}
        f_obj = None

        # Loop over three percentile ranges
        # for col, percentile in enumerate([95, 99, 99.9]):
        for col, percentile in enumerate([99]):
            fig, ax = plt.subplots(1, 1, figsize=(3.4,2.6))    
            # ax = fig.add_subplot(gs[row, col])
            lower, upper = np.percentile(normalized_seps, [0, percentile])
            if lower == upper:
                upper += 1e-6  # Avoid zero-width range

            bins = np.linspace(lower, upper, n_bins)

            # Plot histogram without density normalization
            counts, _ = np.histogram(normalized_seps, bins=bins)
            # Normalize the histogram manually
            bin_widths = np.diff(bins)
            ax.bar(bins[:-1], counts / (len(normalized_seps) * bin_widths), width=bin_widths, alpha=0.7)
            # print(np.sum(counts / (len(normalized_seps) * bin_widths) * bin_widths))

            # ax.hist(normalized_seps, bins=bins, density=True, alpha=0.7, range=(lower, upper))
            ax.set_title(f'Energy Range: [{energy_range[0]:.3f}, {energy_range[1]:.3f}], $N = {normalized_seps.size:,}$\nBins = {n_bins}, $\\langle s \\rangle = {avg_sep:.6f}$', fontsize=10) #(0-{percentile}\%)\n
            ax.set_ylabel('Density')
            ax.set_xlabel('$s/\\langle s \\rangle$')
            # ax.grid(True, alpha=0.3)
            # overlay_gue_curve(ax)
            ax.text(0.9, 0.9, f'{name}', transform=ax.transAxes, fontsize=14, verticalalignment='top',horizontalalignment="right")
            ax.text(0.9, 0.7, f'(0-{percentile}\%)', transform=ax.transAxes, fontsize=14, verticalalignment='top',horizontalalignment="right")

        # Log-scale histogram subplot.
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.6))    
        min_val, max_val = np.min(normalized_seps), np.max(normalized_seps)
        if min_val == max_val:
            max_val += 1e-6

        bins = np.linspace(min_val, max_val, n_bins)
        ax.hist(normalized_seps, bins=bins, density=True, alpha=0.7)
        ax.set_yscale('log')
        ax.set_title(f'Energy Range: [{energy_range[0]:.3f}, {energy_range[1]:.3f}], $N = {normalized_seps.size:,}$\nBins = {n_bins}, $\\langle s \\rangle = {avg_sep:.6f}$', fontsize=10) #(0-{percentile}\%)\n
        ax.set_ylabel('Log Density')
        ax.set_xlabel('$s/\\langle s \\rangle$')
        # ax.grid(True, alpha=0.3, which='both')
        ax.text(0.9, 0.9, f'{name}', transform=ax.transAxes, fontsize=14, verticalalignment='top',horizontalalignment="right")
        # overlay_gue_curve(ax)
        # ax.set_ylim(1e-4, 1.7)

        plt.tight_layout()
        plt.savefig(f'{name}_logctr.pdf')
        
        # Log-log scale histogram subplot
        ax = fig.add_subplot(gs[row, 2])
        min_val, max_val = np.min(normalized_seps), np.max(normalized_seps)
        if min_val == max_val:
            max_val += 1e-6

        bins = np.logspace(np.log10(min_val), np.log10(max_val), n_bins)
        ax.hist(normalized_seps, bins=bins, density=False, alpha=0.7)
        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.set_title(f'{name} - Log-Log Scale (100%)\n$N = {normalized_seps.size:,}$, Bins = {n_bins}\n$\\langle s \\rangle = {avg_sep:.6f}$', fontsize=10)
        ax.set_ylabel('Log Density')
        ax.set_xlabel('Log Normalized Separation $s/\\langle s \\rangle$')
        ax.grid(True, alpha=0.3, which='both')


# analyze_eigenvalue_separations (as before)
def analyze_eigenvalue_separations(folder_path, initial_range=(-0.3, 0.3)):
    valid_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    print(f"Found {len(valid_files)} .npz files in the folder.")

    with PdfPages('eigenvalue_separations_center_1024_xx.pdf') as pdf:
        current_range = np.array(initial_range)
        iteration = 0

        print(f"\nIteration {iteration}: Processing energy range [{current_range[0]:.3f}, {current_range[1]:.3f}]")
        all_separations = {name: [] for name in CHERN_FILTERS}

        for i, fname in enumerate(valid_files, 1):
            if i % 10 == 0:
                print(f"  Processing file {i}/{len(valid_files)}")
            data = np.load(os.path.join(folder_path, fname))
            if not np.isclose(data['SumChernNumbers'], 1, atol=1e-5):
                continue

            trial_data = {
                'eigs': data['eigsPipi'],
                'chern': data['ChernNumbers']
            }
            trial_results = process_trial(trial_data, current_range, CHERN_FILTERS)

            for name, seps in trial_results.items():
                all_separations[name].append(seps)

        all_separations = {k: np.concatenate(v) for k, v in all_separations.items()}
        data_lengths = [len(v) for v in all_separations.values()]
        min_points = min(data_lengths)
        max_points = max(data_lengths)

        print(f"  Minimum data points across all filters: {min_points}")
        print(f"  Maximum data points across all filters: {max_points}")
       
        print("  Generating plots...")
        # generate_plotsKS(current_range, all_separations, pdf, fit_exponential=fit_exponential)
        generate_plots_save(current_range, all_separations, pdf, fit_exponential=fit_exponential)

    print("\nAnalysis complete. PDF file 'eigenvalue_separations_full.pdf' has been generated.")

# Execute analysis (as before)
analyze_eigenvalue_separations(folder_path="/Users/eddiedeleu/Desktop/FinalData/N=2048_MEM",
                               initial_range=(-0.05, 0.05))
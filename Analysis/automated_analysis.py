import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from matplotlib import gridspec, rc
from scipy.optimize import curve_fit

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
    "grid.color": "gray",
    "grid.alpha": 0.6,
    "axes.grid": True,
    "legend.loc": "best",
})
rc('text.latex', preamble=r'\usepackage{amsmath}')

# Chern filters (as before)
CHERN_FILTERS = {
    r'All Chern': None,
    r'Chern $0$ or $+1$': [0, 1],
    r'Chern $0$ or $-1$': [0, -1],
    r'Only Chern $0$': [0],
    r'Only Chern $-1$': [-1],
    r'Only Chern $+1$': [1],
    r'Magnitude=1 Chern': [-1, 1],
    r'Non-Zero Chern': [-3,-2,-1, 1,2,3]
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

def get_dynamic_bin_count(n_points):
    if n_points >= 50000:
        return 500
    elif n_points >= 10000:
        return int(100 + (n_points - 10000) * 400 / 40000)
    elif n_points >= 5000:
        return int(50 + (n_points - 5000) * 50 / 5000)
    else:
        return max(20, int(n_points / 100))

def exponential_function(x, lam):
    return lam * np.exp(-lam * x)

def generate_plots(energy_range, all_separations, pdf, fit_exponential=False):
    fig = plt.figure(figsize=(20, 5 * len(CHERN_FILTERS)))
    gs = gridspec.GridSpec(len(CHERN_FILTERS), 4, figure=fig)
    fig.suptitle(f"Energy Range: [{energy_range[0]:.3f}, {energy_range[1]:.3f}]", y=0.98, fontsize=16)

    for row, (name, seps) in enumerate(all_separations.items()):
        if seps.size == 0:
            continue

        normalized_seps, avg_sep = normalize_separations(seps)
        n_bins = get_dynamic_bin_count(normalized_seps.size)

        # Exponential fit (as before)
        popt = None
        if fit_exponential and name in [r'Only Chern $0$', r'Only Chern $-1$', r'Only Chern $+1$'] and normalized_seps.size > 10:
            try:
                popt, pcov = curve_fit(exponential_function, normalized_seps, np.ones_like(normalized_seps), p0=[1])
            except RuntimeError:
                print(f"Curve fit failed for {name}")

        for col, percentile in enumerate([95, 99, 99.9]):
            ax = fig.add_subplot(gs[row, col])
            lower, upper = np.percentile(normalized_seps, [0, percentile])

            # Handle edge case: If lower and upper are the same, add a small offset
            if lower == upper:
                upper += 1e-6  # Add a tiny offset

            bins = np.linspace(lower, upper, n_bins)
            ax.hist(normalized_seps, bins=bins, density=True, alpha=0.7, range=(lower, upper))
            ax.set_title(f'{name} (0-{percentile}%)\n$N = {normalized_seps.size:,}$, Bins = {n_bins}\n$\\langle s \\rangle = {avg_sep:.4f}$', fontsize=10)
            ax.set_ylabel(r'Density')
            ax.set_xlabel(r'Normalized Separation $s/\langle s \rangle$')
            ax.grid(True, alpha=0.3)

            # Plot exponential fit (as before)
            if popt is not None:
                x = np.linspace(lower, upper, 100)
                ax.plot(x, exponential_function(x, *popt), 'r-', label=f'Exp. Fit: $\\lambda$={popt[0]:.2f}')
                ax.legend()

        ax = fig.add_subplot(gs[row, 3])
        # Handle edge case: If min and max are the same, add a small offset
        min_val = np.min(normalized_seps)
        max_val = np.max(normalized_seps)
        if min_val == max_val:
            max_val += 1e-6

        bins = np.linspace(min_val, max_val, n_bins)
        ax.hist(normalized_seps, bins=bins, density=True, alpha=0.7)
        ax.set_yscale('log')
        ax.set_title(f'{name} - Log Scale (100%)\n$N = {normalized_seps.size:,}$, Bins = {n_bins}\n$\\langle s \\rangle = {avg_sep:.4f}$', fontsize=10)
        ax.set_ylabel(r'Log Density')
        ax.set_xlabel(r'Normalized Separation $s/\langle s \rangle$')
        ax.grid(True, alpha=0.3, which='both')

        # Plot exponential fit (as before)
        if popt is not None:
            x = np.linspace(min_val, max_val, 100)
            ax.plot(x, exponential_function(x, *popt), 'r-', label=f'Exp. Fit: $\\lambda$={popt[0]:.2f}')
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    pdf.savefig()
    plt.close()

# analyze_eigenvalue_separations (as before)
def analyze_eigenvalue_separations(folder_path, initial_range=(-0.3, 0.3), min_data=10000, halt_on_max=False, fit_exponential=False):
    valid_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    print(f"Found {len(valid_files)} .npz files in the folder.")

    with PdfPages('eigenvalue_separations.pdf') as pdf:
        current_range = np.array(initial_range)
        iteration = 0

        while True:
            iteration += 1
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
            
            if halt_on_max:
                halt_condition = max_points < min_data
                print("  Halting when maximum data points is less than threshold.")
            else:
                halt_condition = min_points < min_data
                print("  Halting when minimum data points is less than threshold.")

            if halt_condition:
                print(f"Stopping at range [{current_range[0]:.3f}, {current_range[1]:.3f}] with minimum {min_points} and maximum {max_points} points")
                break

            print("  Generating plots...")
            generate_plots(current_range, all_separations, pdf, fit_exponential=fit_exponential)
            current_range *= 0.7

    print("\nAnalysis complete. PDF file 'eigenvalue_separations.pdf' has been generated.")

# Execute analysis (as before)
analyze_eigenvalue_separations(folder_path="/Users/eddiedeleu/Desktop/FinalData/N=1024_MEM",
                               initial_range=(-0.3, 0.3),
                               min_data=6000,
                               halt_on_max=False,
                               fit_exponential=True)

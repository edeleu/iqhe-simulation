import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from matplotlib import gridspec, rc

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
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

CHERN_FILTERS = {
    r'All Chern': None,
    r'C = $0$ or $+1$': [0, 1],
    r'C = $0$ or $-1$': [0, -1],
    r'C = $0$': [0],
    r'C = $-1$': [-1],
    r'C = $+1$': [1],
    r'$|$C$|$ = 1': [-1, 1],
    r'C $\\ne 0$': [-3, -2, -1, 1, 2, 3]
}

POWER_LAWS = [
    (2, r'$x^2$'),
    (2.5, r'$x^{2.5}$'),
    (3, r'$x^{3}$')
]

BOUNDARY_KEYS = ["eigs00", "eigs0pi", "eigsPi0", "eigsPipi"]


def process_trial(data, energy_range, chern_filters):
    eigs = data['eigs']
    chern = data['chern']
    mask = (eigs >= energy_range[0]) & (eigs <= energy_range[1])
    trial_eigs = eigs[mask]
    trial_chern = chern[mask]
    results = {}
    for name, cf in chern_filters.items():
        filtered = trial_eigs if cf is None else trial_eigs[np.isin(trial_chern, cf)]
        seps = np.diff(np.sort(filtered)) if len(filtered) > 1 else np.array([])
        results[name] = seps
    return results


def normalize_separations(seps):
    if len(seps) == 0:
        return seps, 0
    mean_sep = np.mean(seps)
    return seps / mean_sep, mean_sep


def get_dynamic_bin_count(n):
    A, alpha = 3.25, 0.31
    bins = int(np.ceil(A * n**alpha))
    return min(bins, 250)


def regroup_trials_by_file(trial_files, chern_filters, energy_range):
    grouped_separations = {name: [[] for _ in BOUNDARY_KEYS] for name in chern_filters}

    for fname in trial_files:
        data = np.load(fname)
        if not np.isclose(data['SumChernNumbers'], 1, atol=1e-5):
            continue
        for idx, key in enumerate(BOUNDARY_KEYS):
            trial_data = {'eigs': data[key], 'chern': data['ChernNumbers']}
            result = process_trial(trial_data, energy_range, chern_filters)
            for label, seps in result.items():
                if len(seps) > 0:
                    grouped_separations[label][idx].append(seps)

    for label in grouped_separations:
        for i in range(len(BOUNDARY_KEYS)):
            grouped_separations[label][i] = np.concatenate(grouped_separations[label][i]) if grouped_separations[label][i] else np.array([])

    return grouped_separations


def generate_scatter_histograms(grouped_separations, energy_range, pdf):
    for name, group_list in grouped_separations.items():
        if all(len(g) < 2 for g in group_list):
            continue

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f"{name}  |  Energy: [{energy_range[0]:.3f}, {energy_range[1]:.3f}]", fontsize=12)

        for i, grp in enumerate(group_list):
            norm_grp, _ = normalize_separations(grp)
            n_bins = get_dynamic_bin_count(len(norm_grp))

            cutoff = np.percentile(norm_grp, 99)
            data99 = norm_grp[norm_grp <= cutoff]
            if len(data99) > 1:
                bins99 = np.linspace(data99.min(), data99.max(), n_bins)
                counts99, edges99 = np.histogram(data99, bins=bins99)
                widths99 = np.diff(edges99)
                centers99 = edges99[:-1] + widths99 / 2
                density99 = counts99 / (len(norm_grp) * widths99)
                axs[0].scatter(centers99, density99, s=4, color=color_cycle[i])

            axs[0].set_title(f'Linear (99\%)\nN={len(grp)}, bins={n_bins}', fontsize=10)
            axs[0].set_xlabel(r"$s/\langle s \rangle$")
            axs[0].set_ylabel(r"$P(s/\langle s \rangle)$")

            bins100 = np.linspace(norm_grp.min(), norm_grp.max(), n_bins)
            counts100, edges100 = np.histogram(norm_grp, bins=bins100)
            widths100 = np.diff(edges100)
            centers100 = edges100[:-1] + widths100 / 2
            density100 = counts100 / (len(norm_grp) * widths100)
            axs[1].scatter(centers100, density100, s=4, color=color_cycle[i])
            axs[1].set_title(f'Linear (100\%)\nN={len(grp)}, bins={n_bins}', fontsize=10)
            axs[1].set_xlabel(r"$s/\langle s \rangle$")
            axs[1].set_ylabel(r"$P(s/\langle s \rangle)$")

            axs[2].semilogy(centers100, density100, marker='.', linestyle='None', markersize=4,
                           color=color_cycle[i], label=f'G{i+1}: {len(grp)} pts')
            axs[2].set_title(f"Log-Linear\nN={len(grp)}, bins={n_bins}", fontsize=10)
            axs[2].set_xlabel(r"$s/\langle s \rangle$")
            axs[2].set_ylabel(r"$\log P(s/\langle s \rangle)$")
            axs[2].legend(fontsize=6)

            pos = norm_grp[norm_grp > 0]
            if len(pos) > 1:
                bins_log = np.logspace(np.log10(pos.min()), np.log10(pos.max()), n_bins)
                counts_log, edges_log = np.histogram(pos, bins=bins_log)
                centers_log = edges_log[:-1] * np.sqrt(edges_log[1:] / edges_log[:-1])
                axs[3].loglog(centers100, density100, marker='.', linestyle='None', markersize=4,
                             color=color_cycle[i])

        axs[3].set_title(f"Log-Log\nBins={n_bins}", fontsize=10)
        axs[3].set_xlabel(r"$\log s/\langle s \rangle$")
        axs[3].set_ylabel(r"$\log P(s/\langle s \rangle)$")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

def analyze_folder(folder_path, energy_range=(-0.3, 0.3)):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npz')]
    total = len(files)
    print(f"Processing {total} files.")

    grouped_separations = regroup_trials_by_file(files, CHERN_FILTERS, energy_range)

    with PdfPages('ps_scatter_plotsTrial.pdf') as pdf:
        generate_scatter_histograms(grouped_separations, energy_range, pdf)


analyze_folder("/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/N=1024_Mem/", energy_range=(-0.03, 0.03))

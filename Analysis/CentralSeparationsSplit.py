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
# Color cycle for subgroups
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Chern filters
CHERN_FILTERS = {
    r'All Chern': None,
    r'C = $0$ or $+1$': [0, 1],
    r'C = $0$ or $-1$': [0, -1],
    r'C = $0$': [0],
    r'C = $-1$': [-1],
    r'C = $+1$': [1],
    r'$|$C$|$ = 1': [-1, 1],
    r'C $\ne 0$': [-3, -2, -1, 1, 2, 3]
}

# -- Custom Distributions for Wigner Surmise --
def overlay_gue_curve(ax, s_max=6, num_points=1000, label="GUE", color="green", linestyle="--"):
    s = np.linspace(0, s_max, num_points)
    p_s = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
    # p_s = np.exp(-1.65*(s-0.6))
    # p_s = s*np.exp(-s)

    ax.plot(s, p_s, label=label, color=color, linestyle=linestyle)

# Data processing
def process_trial(data, energy_range, chern_filters):
    eigs = data['eigs']
    chern = data['chern']
    mask = (eigs >= energy_range[0]) & (eigs <= energy_range[1])
    trial_eigs = eigs[mask]
    trial_chern = chern[mask]
    results = {}
    for name, cf in chern_filters.items():
        if cf is None:
            filtered = trial_eigs
        else:
            filtered = trial_eigs[np.isin(trial_chern, cf)]
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


def split_into_groups(seps, n_groups=1):
    seps = np.asarray(seps)
    np.random.shuffle(seps)
    return [seps[i::n_groups] for i in range(n_groups)]

# Plotting

def generate_scatter_histograms(all_separations, energy_range, pdf):
    # Define power-law overlays (exponent, label)
    POWER_LAWS = [
        (2, r'$x^2$'),
        (2.5, r'$x^{2.5}$'),
        (3, r'$x^{3}$')
    ]

    # Custom scaling factors per Chern filter
    POWER_LAW_SCALES = {
        r'All Chern': {2: 2e4, 2.5: 3e4, 3: 1e4},
        r'C = $0$ or $+1$': {2: 3e4, 2.5: 3e4, 3: 1e4},
        r'C = $0$ or $-1$': {2: 3e4, 2.5: 3e4, 3: 1e4},
        r'C = $0$': {2: 3e4, 2.5: 3e4, 3: 1e4},
        r'C = $-1$': {2: 8e3, 2.5: 8e3, 3: 1e4},
        r'C = $+1$': {2: 8e3, 2.5: 8e3, 3: 1e4},
        r'$|$C$|$ = 1': {2: 3e4, 2.5: 3e4, 3: 2e4},
        r'C $\ne 0$': {2: 3e4, 2.5: 3e4, 3: 2e4},
    }

    for name, seps in all_separations.items():
        if len(seps) < 2:
            continue

        groups = split_into_groups(seps)
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(
            f"{name}  |  Energy: [{energy_range[0]:.3f}, {energy_range[1]:.3f}]",
            fontsize=12
        )

        all_centers_log = []
        all_counts_log = []

        for i, grp in enumerate(groups):
            norm_grp, _ = normalize_separations(grp)
            n_bins = get_dynamic_bin_count(len(norm_grp))

            # Linear (99%)
            cutoff = np.percentile(norm_grp, 99)
            data99 = norm_grp[norm_grp <= cutoff]
            if len(data99) > 1:
                bins99 = np.linspace(data99.min(), data99.max(), n_bins)
                counts99, edges99 = np.histogram(data99, bins=bins99)
                widths99 = np.diff(edges99)
                centers99 = edges99[:-1] + widths99 / 2
                density99 = counts99 / (len(norm_grp) * widths99)
                axs[0].scatter(
                    centers99, density99, s=4, color=color_cycle[i]
                )
            axs[0].set_title(
                f'Linear (99\%)\nN={len(seps)}, bins={n_bins}',
                fontsize=10
            )
            axs[0].set_xlabel(r"$s/\langle s \rangle$")
            axs[0].set_ylabel(r"$P(s/\langle s \rangle)$")

            # Linear (100%)
            bins100 = np.linspace(norm_grp.min(), norm_grp.max(), n_bins)
            counts100, edges100 = np.histogram(norm_grp, bins=bins100)
            widths100 = np.diff(edges100)
            centers100 = edges100[:-1] + widths100 / 2
            density100 = counts100 / (len(norm_grp) * widths100)
            axs[1].scatter(
                centers100, density100, s=4, color=color_cycle[i]
            )
            axs[1].set_title(
                f'Linear (100\%)\nN={len(seps)}, bins={n_bins}',
                fontsize=10
            )
            axs[1].set_xlabel(r"$s/\langle s \rangle$")
            axs[1].set_ylabel(r"$P(s/\langle s \rangle)$")

            # Log-Linear
            axs[2].semilogy(
                centers100, density100,
                marker='.', linestyle='None', markersize=4,
                color=color_cycle[i],
                label=f'G{i+1}: {len(norm_grp)} pts'
            )
            axs[2].set_title(
                f"Log-Linear\nN={len(seps)}, bins={n_bins}",
                fontsize=10
            )
            axs[2].set_xlabel(r"$s/\langle s \rangle$")
            axs[2].set_ylabel(r"$\log P(s/\langle s \rangle)$")
            axs[2].legend(fontsize=6)
            # overlay_gue_curve(axs[2])
            # axs[2].set_ylim(1e-4, 1.7)

            # Log-Log
            pos = norm_grp[norm_grp > 0]
            if len(pos) > 1:
                bins_log = np.logspace(
                    np.log10(pos.min()), np.log10(pos.max()), n_bins
                )
                counts_log, edges_log = np.histogram(pos, bins=bins_log)
                centers_log = edges_log[:-1] * np.sqrt(edges_log[1:] / edges_log[:-1])
                axs[3].loglog(
                    centers_log, counts_log,
                    marker='.', linestyle='None', markersize=4,
                    color=color_cycle[i]
                )
                all_centers_log.extend(centers_log)
                all_counts_log.extend(counts_log)

        axs[3].set_title(
            f"Log-Log\nN={len(seps)}, bins={n_bins}",
            fontsize=10
        )
        axs[3].set_xlabel(r"$\log s/\langle s \rangle$")
        axs[3].set_ylabel(r"$\log P(s/\langle s \rangle)$")

        # === Overlay power laws ONCE per filter ===
        if all_centers_log:
            all_centers_log = np.array(all_centers_log)
            x_min = np.min(all_centers_log)
            x_max = np.max(all_centers_log)
            x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 500)
            scales = POWER_LAW_SCALES.get(name, {})
            for exp, label in POWER_LAWS:
                scale = scales.get(exp, None)
                if scale is not None:
                    y_fit = scale * x_fit**exp
                    axs[3].loglog(x_fit, y_fit, linestyle='--', linewidth=1, label=label)
            axs[3].legend(fontsize=6)
            y_all = np.array(all_counts_log)
            y_positive = y_all[y_all > 0]
            if len(y_positive) > 0:
                ymin, ymax = y_positive.min(), y_positive.max()
                axs[3].set_ylim(0.8 * ymin, 1.2 * ymax)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

def analyze_folder(folder_path, energy_range=(-0.3, 0.3)):
    files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    total = len(files)
    print(f"Processing {total} files.")
    all_seps = {name: [] for name in CHERN_FILTERS}
    for idx, f in enumerate(files, 1):
        if idx % 100 == 0:
            print(f"  Processed {idx}/{total} files.")
        data = np.load(os.path.join(folder_path, f))
        if not np.isclose(data['SumChernNumbers'], 1, atol=1e-5):
            continue
        trial = {'eigs': data['eigsPipi'], 'chern': data['ChernNumbers']}
        res = process_trial(trial, energy_range, CHERN_FILTERS)
        for name, seps in res.items():
            if len(seps):
                all_seps[name].append(seps)

    all_seps = {k: np.concatenate(v) for k, v in all_seps.items() if v}
    with PdfPages('ps_scatter_plots.pdf') as pdf:
        generate_scatter_histograms(all_seps, energy_range, pdf)

analyze_folder("/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/N=1024_Mem/", energy_range=(-0.03, 0.03))

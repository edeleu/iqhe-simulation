import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from matplotlib import gridspec, rc
import scipy.stats as stats
import numpy as np
from scipy.special import gamma
from scipy.optimize import minimize_scalar
from scipy.optimize import curve_fit
from scipy.stats import kstest
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import interp1d

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

def gue_pdf(s):
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)

# Numerical CDF of the GUE PDF
def gue_cdf(s_vals, b=2):
    if b==2:
        pdf_vals = gue_pdf(s_vals)
    else:
        pdf_vals = normalized_pdf(s_vals,b)

    cdf_vals = cumtrapz(pdf_vals, s_vals, initial=0)
    return cdf_vals / cdf_vals[-1]  # normalize to 1

# Perform KS test
def perform_ks_test(norm_all,b=2):
    s_vals = np.linspace(1e-3, np.max(norm_all) * 1.1, 1000)

    # Interpolated CDF of model
    cdf_model_vals = gue_cdf(s_vals,b)

    # Interpolated function for use with scipy's kstest
    model_cdf_interp = interp1d(s_vals, cdf_model_vals, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))

    # Run the KS test
    stat, p_value = kstest(norm_all, model_cdf_interp)
    return stat, p_value

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

        #sorted_filtered = np.sort(filtered)
        #seps = sorted_filtered[2:] - sorted_filtered[:-2] if len(sorted_filtered) > 2 else np.array([])

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

# --- PDF utilities ---
def c_of_b(b):
    return (gamma((b + 2) / 2) / gamma((b + 1) / 2))**2

def normalized_pdf(s, b):
    if b <= -1 or np.any(s <= 0):
        return np.zeros_like(s)
    c = c_of_b(b)
    A = 2 * c**((b + 1) / 2) / gamma((b + 1) / 2)
    return A * s**b * np.exp(-c * s**2)

def hist_pdf(s, b):
    return normalized_pdf(s, b)

def neg_log_likelihood(b, s_data):
    if b <= -1:
        return np.inf
    pdf_vals = normalized_pdf(s_data, b)
    if np.any(pdf_vals <= 0):
        return np.inf
    return -np.sum(np.log(pdf_vals))

def generate_scatter_histograms(all_separations, energy_range, pdf):

    for name, seps in all_separations.items():
        if len(seps) < 2:
            continue

        norm_all, _ = normalize_separations(seps)
        result = minimize_scalar(neg_log_likelihood, bounds=(0.5, 8), args=(norm_all,), method='bounded')
        b_fit = result.x

        s_fit = np.linspace(1e-2, np.max(norm_all) * 1.05, 400)
        pdf_fit = normalized_pdf(s_fit, b_fit)

        ks_stat, ks_p = perform_ks_test(norm_all,b_fit)
        print(f"KS statistic: {ks_stat:.4f}, p-value: {ks_p:.4f}")

        groups = split_into_groups(seps)
        fig, axs = plt.subplots(1, 5, figsize=(22, 4))
        fig.suptitle(
            f"{name}  |  Energy: [{energy_range[0]:.3f}, {energy_range[1]:.3f}]  |  Fitted $b = {b_fit:.3f}$",
            fontsize=12)

        for i, grp in enumerate(groups):
            norm_grp, _ = normalize_separations(grp)
            n_bins = get_dynamic_bin_count(len(norm_grp))

            # Linear 99% plot
            cutoff = np.percentile(norm_grp, 99)
            data99 = norm_grp[norm_grp <= cutoff]
            if len(data99) > 1:
                bins99 = np.linspace(data99.min(), data99.max(), n_bins)
                counts99, edges99 = np.histogram(data99, bins=bins99)
                widths99 = np.diff(edges99)
                centers99 = edges99[:-1] + widths99 / 2
                density99 = counts99 / (len(norm_grp) * widths99)
                axs[0].scatter(centers99, density99, s=4, color=color_cycle[i])
                axs[0].plot(s_fit, normalized_pdf(s_fit, b_fit), 'r--', label=f"Fit b={b_fit:.2f}")

            axs[0].set_title(f'Linear (99\%)\nN={len(seps)}, bins={n_bins}', fontsize=10)
            axs[0].set_xlabel(r"$s/\langle s \rangle$")
            axs[0].set_ylabel(r"$P(s/\langle s \rangle)$")

            # Linear 100% plot
            bins100 = np.linspace(norm_grp.min(), norm_grp.max(), n_bins)
            counts100, edges100 = np.histogram(norm_grp, bins=bins100)
            widths100 = np.diff(edges100)
            centers100 = edges100[:-1] + widths100 / 2
            density100 = counts100 / (len(norm_grp) * widths100)
            axs[1].scatter(centers100, density100, s=4, color=color_cycle[i])
            axs[1].plot(s_fit, normalized_pdf(s_fit, b_fit), 'r--')
            

            axs[1].set_title(f'Linear (100\%)\nN={len(seps)}, bins={n_bins}', fontsize=10)
            axs[1].set_xlabel(r"$s/\langle s \rangle$")
            axs[1].set_ylabel(r"$P(s/\langle s \rangle)$")

            # Log-linear
            axs[2].semilogy(centers100, density100, marker='.', linestyle='None', markersize=4,
                            color=color_cycle[i], label=f'G{i+1}: {len(norm_grp)} pts')
            axs[2].plot(s_fit, normalized_pdf(s_fit, b_fit), 'r--')
            axs[2].set_title(f"Log-Linear\nN={len(seps)}, bins={n_bins}", fontsize=10)
            axs[2].set_xlabel(r"$s/\langle s \rangle$")
            axs[2].set_ylabel(r"$\log P(s/\langle s \rangle)$")
            axs[2].legend(fontsize=6)
            axs[2].set_ylim(1e-4, 1.7)

            # Fit b by minimizing squared error to histogram
            popt, _ = curve_fit(hist_pdf, centers100, density100, bounds=(0.5, 10))
            b_fit_hist = popt[0]
            print(f"Best-fit b (histogram fitted): {b_fit_hist:.4f}")

            # Log-log
            pos = norm_grp[norm_grp > 0]
            if len(pos) > 1:
                bins_log = np.logspace(np.log10(pos.min()), np.log10(pos.max()), n_bins)
                counts_log, edges_log = np.histogram(pos, bins=bins_log)
                centers_log = edges_log[:-1] * np.sqrt(edges_log[1:] / edges_log[:-1])
                axs[3].loglog(centers100, density100, marker='.', linestyle='None', markersize=4,
                              color=color_cycle[i])
                axs[3].plot(s_fit, normalized_pdf(s_fit, b_fit), 'r--')

            # === Q-Q Plot vs GUE: decimated + percentile quantiles ===

            # ---- (1) Decimated plot: uniform spacing in index space ----
            norm_sorted = np.sort(norm_grp)
            N = len(norm_sorted)
            probs_uniform = (np.arange(1, N + 1) - 0.5) / N
            s_grid = np.linspace(1e-3, max(5, norm_sorted[-1] * 1.1), 2000)
            cdf_gue = gue_cdf(s_grid,b=b_fit)
            inv_cdf_gue = interp1d(cdf_gue, s_grid, kind='linear', bounds_error=False, fill_value='extrapolate')
            gue_quantiles_uniform = inv_cdf_gue(probs_uniform)

            step = max(1, len(norm_sorted) // 10000)
            axs[4].plot(gue_quantiles_uniform[::step], norm_sorted[::step], '.', markersize=1.5, alpha=0.6, label='Sampled 10k')

            # ---- (2) Quantile-based plot: 100 percentiles ----
            quantile_probs = np.linspace(0.01, 0.99, 200)
            emp_q = np.quantile(norm_grp, quantile_probs)
            gue_q = inv_cdf_gue(quantile_probs)
            axs[4].plot(gue_q, emp_q, 'o', markersize=1.5, alpha=0.9, label='200 Quantiles')

            # ---- Ideal reference line ----
            lims = [0, max(gue_q.max(), emp_q.max(), norm_sorted.max())]
            axs[4].plot(lims, lims, 'k--', lw=0.8, label='Ideal')

            # ---- Styling ----
            axs[4].set_title("Q-Q Plot vs GUE", fontsize=10)
            axs[4].set_xlabel("GUE Quantiles")
            axs[4].set_ylabel("Empirical Quantiles")
            axs[4].legend(fontsize=6, loc='upper left', frameon=False)
            axs[4].grid(True, alpha=0.3)

        axs[3].set_title(f"Log-Log\nN={len(seps)}, bins={n_bins}", fontsize=10)
        axs[3].set_xlabel(r"$\log s/\langle s \rangle$")
        axs[3].set_ylabel(r"$\log P(s/\langle s \rangle)$")
        # axs[1].set_ylim(0, 1.05)
        axs[2].set_ylim(5e-5, 1.7)
        axs[3].set_ylim(5e-5, 1.7)

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

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
from scipy.optimize import minimize

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
    # r'C = $0$ or $+1$': [0, 1],
    # r'C = $0$ or $-1$': [0, -1],
    # r'C = $0$': [0],
    # r'C = $-1$': [-1],
    # r'C = $+1$': [1],
    # r'$|$C$|$ = 1': [-1, 1],
    # r'C $\ne 0$': [-3, -2, -1, 1, 2, 3]
}

def poly_model(x, A, B):
    return A * x**2 + B * x**4
def power_law(x, A, beta):
    return A * x**beta

# -- Custom Distributions for Wigner Surmise --
def overlay_gue_curve(ax, s_max=6, num_points=1000, label="Reference GUE", color="green", linestyle="--"):
    s = np.linspace(0, s_max, num_points)
    p_s = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
    # p_s = np.exp(-1.65*(s-0.6))
    # p_s = s*np.exp(-s)
    # ax.plot(s, p_s, label=label, color=color, linestyle=linestyle)

def gue_pdf(s):
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)


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


# Updated process_trial function implementing three parity scenarios explicitly
def process_trial_three_scenarios(data, energy_range):
    eigs = data['eigs']
    chern = data['chern']
    mask = (eigs >= energy_range[0]) & (eigs <= energy_range[1])
    trial_eigs = eigs[mask]
    trial_chern = chern[mask]

    sort_indices = np.argsort(trial_eigs)
    sorted_eigs = trial_eigs[sort_indices]
    sorted_chern = trial_chern[sort_indices]

    results = {
        'C=+1 to nearest -1': [],
        'C=-1 to nearest +1': [],
        'C=±1 to nearest opposite parity': []
    }

    for i in range(len(sorted_eigs)):
        current_chern = sorted_chern[i]
        if current_chern not in [-1, 1]:
            continue

        # Scenario: Current C=+1, nearest C=-1
        if current_chern == 1:
            j = i - 1
            nearest_minus_dist = np.inf
            while j >= 0:
                if sorted_chern[j] == -1:
                    nearest_minus_dist = sorted_eigs[i] - sorted_eigs[j]
                    break
                j -= 1
            j = i + 1
            while j < len(sorted_eigs):
                if sorted_chern[j] == -1:
                    dist_right = sorted_eigs[j] - sorted_eigs[i]
                    nearest_minus_dist = min(nearest_minus_dist, dist_right)
                    break
                j += 1
            if nearest_minus_dist < np.inf:
                results['C=+1 to nearest -1'].append(nearest_minus_dist)

        # Scenario: Current C=-1, nearest C=+1
        if current_chern == -1:
            j = i - 1
            nearest_plus_dist = np.inf
            while j >= 0:
                if sorted_chern[j] == 1:
                    nearest_plus_dist = sorted_eigs[i] - sorted_eigs[j]
                    break
                j -= 1
            j = i + 1
            while j < len(sorted_eigs):
                if sorted_chern[j] == 1:
                    dist_right = sorted_eigs[j] - sorted_eigs[i]
                    nearest_plus_dist = min(nearest_plus_dist, dist_right)
                    break
                j += 1
            if nearest_plus_dist < np.inf:
                results['C=-1 to nearest +1'].append(nearest_plus_dist)

        # Scenario: C=±1 to nearest opposite parity
        j = i - 1
        nearest_opposite_dist = np.inf
        while j >= 0:
            if sorted_chern[j] == -current_chern:
                nearest_opposite_dist = sorted_eigs[i] - sorted_eigs[j]
                break
            j -= 1
        j = i + 1
        while j < len(sorted_eigs):
            if sorted_chern[j] == -current_chern:
                dist_right = sorted_eigs[j] - sorted_eigs[i]
                nearest_opposite_dist = min(nearest_opposite_dist, dist_right)
                break
            j += 1
        if nearest_opposite_dist < np.inf:
            results['C=±1 to nearest opposite parity'].append(nearest_opposite_dist)

    return {k: np.array(v) for k, v in results.items()}


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
def norm_coeffs(n,y):
    """Returns A,B such that the distribution has unit mean and is a proper PDF."""
    gamma1 = gamma((1+n) / y)
    gamma2 = gamma((2+n) / y)

    B = (gamma2/gamma1)**(y)
    A = (y*B**((1+n)/y)) / gamma1

    return A, B

def normalized_pdf(s, n, y):
    """
    PDF of the normalized distribution:
    f(s; n, y) = A(n,y) * s^n * exp(-B(n,y) * s^y)
    where A and B ensure normalization and unit mean.

    s, y >= 0, n >= -1
    """
    
    A, B = norm_coeffs(n, y)
    return A * s**n * np.exp(-B * s**y)

def hist_pdf(s, *params, fixed=None):
    """Used for curve_fit to histogram."""
    if fixed is None:
        n, y = params
    elif 'y' in fixed:
        n, y = params[0], fixed['y']
    elif 'n' in fixed:
        n, y = fixed['n'], params[0]
    return normalized_pdf(s, n, y)

def neg_log_likelihood(params, s_data, fixed=None):
    """params: array of params to be fit; fixed: dict like {'n': 2} or {'y': 2}"""
    if fixed is None:
        n, y = params
    elif 'y' in fixed:
        n, y = params[0], fixed['y']
    elif 'n' in fixed:
        n, y = fixed['n'], params[0]

    if n <= -1 or y <= 0:
        return np.inf
    pdf_vals = normalized_pdf(s_data, n, y)
    if np.any(pdf_vals <= 0):
        return np.inf
    return -np.sum(np.log(pdf_vals))

def compute_chi_squared(norm_data, model_pdf_func, params=None, num_bins=50):
    counts, bin_edges = np.histogram(norm_data, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)
    total_counts = np.sum(counts)

    if params is not None:
        model_pdf = model_pdf_func(bin_centers, *params)
        dof = np.sum(model_pdf > 0) - len(params)
    else:
        model_pdf = model_pdf_func(bin_centers)
        dof = np.sum(model_pdf > 0) - 1

    expected = model_pdf * bin_widths * total_counts
    mask = expected > 0
    chi2 = np.sum((counts[mask] - expected[mask])**2 / expected[mask])

    return chi2, dof, chi2 / dof

def gue_cdf(s_vals, n=2, y=2):
    pdf_vals = normalized_pdf(s_vals, n, y)
    cdf_vals = cumtrapz(pdf_vals, s_vals, initial=0)
    return cdf_vals / cdf_vals[-1]

def perform_ks_test(norm_all, n=2, y=2):
    s_vals = np.linspace(1e-3, np.max(norm_all) * 1.1, 1000)
    cdf_model_vals = gue_cdf(s_vals, n, y)
    model_cdf_interp = interp1d(s_vals, cdf_model_vals, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))
    stat, p_value = kstest(norm_all, model_cdf_interp)
    return stat, p_value


def get_all_fits(norm_all):
    # === NLL Fits ===
    result1 = minimize_scalar(lambda n: neg_log_likelihood([n], norm_all, fixed={'y': 2}),
                              bounds=(0.4, 4), method='bounded')
    n1, y1 = result1.x, 2

    result2 = minimize_scalar(lambda y: neg_log_likelihood([y], norm_all, fixed={'n': 2}),
                              bounds=(0.4, 4), method='bounded')
    n2, y2 = 2, result2.x

    result3 = minimize(lambda p: neg_log_likelihood(p, norm_all),
                       x0=[2.0, 2.0], bounds=[(0.4, 4), (0.4, 4)])
    n3, y3 = result3.x

    fits_nll = [(n1, y1), (n2, y2), (n3, y3)]
    print("\n=== Negative Log-Likelihood Fits ===")
    for i, (n, y) in enumerate(fits_nll, 1):
        ks_stat, ks_p = perform_ks_test(norm_all, n, y)
        chi2, dof, red_chi2 = compute_chi_squared(norm_all, normalized_pdf, params=(n, y))
        print(f"Case {i}: n={n:.4f}, y={y:.4f} | KS={ks_stat:.4f} (p={ks_p:.4f}), Chi²={chi2:.2f}, RedChi²={red_chi2:.2f}")

    return fits_nll

def get_all_hist_fits(norm_all):
    counts, bin_edges = np.histogram(norm_all, bins=50)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    widths = np.diff(bin_edges)
    density = counts / (np.sum(counts) * widths)

    popt1, _ = curve_fit(lambda s, n: hist_pdf(s, n, fixed={'y': 2}), centers, density, bounds=(0.5,4))
    popt2, _ = curve_fit(lambda s, y: hist_pdf(s, y, fixed={'n': 2}), centers, density, bounds=(0.5,4))
    popt3, _ = curve_fit(lambda s, n, y: hist_pdf(s, n, y), centers, density, bounds=([0.4, 0.4], [4, 4]))

    fits_hist = [(popt1[0], 2), (2, popt2[0]), (popt3[0], popt3[1])]
    print("\n=== Histogram Curve Fits ===")
    for i, (n, y) in enumerate(fits_hist, 1):
        ks_stat, ks_p = perform_ks_test(norm_all, n, y)
        chi2, dof, red_chi2 = compute_chi_squared(norm_all, normalized_pdf, params=(n, y))
        print(f"Hist Case {i}: n={n:.4f}, y={y:.4f} | KS={ks_stat:.4f} (p={ks_p:.4f}), Chi²={chi2:.2f}, RedChi²={red_chi2:.2f}")

    return fits_hist, centers, density

def generate_scatter_histograms(all_separations, energy_range, pdf):
    for name, seps in all_separations.items():
        if len(seps) < 2:
            continue

        norm_all, _ = normalize_separations(seps)

        fits_nll = get_all_fits(norm_all)
        fits_hist, centers, density = get_all_hist_fits(norm_all)

        # Generate s range
        s_fit = np.linspace(1e-2, np.max(norm_all) * 1.05, 400)

        groups = split_into_groups(seps)
        fig, axs = plt.subplots(1, 5, figsize=(24, 4))
        fig.suptitle(
            f"{name} | Energy: [{energy_range[0]:.3f}, {energy_range[1]:.3f}] | "
            f"NLL Fit: n = {fits_nll[2][0]:.3f}, y = {fits_nll[2][1]:.3f}",
            fontsize=12)

        for i, grp in enumerate(groups):
            norm_grp, _ = normalize_separations(grp)
            n_bins = get_dynamic_bin_count(len(norm_grp))


            binszoom = np.linspace(0, 0.5, 50)
            countszoom, edgeszoom = np.histogram(norm_grp[norm_grp <= 0.5], bins=binszoom)
            widthszoom = np.diff(edgeszoom)
            centerszoom = edgeszoom[:-1] + widthszoom / 2
            densityzoom = countszoom / (len(norm_grp) * widthszoom)
            axs[0].scatter(centerszoom, densityzoom, s=4, color=color_cycle[i], label='Data')

            for j, (n, y) in enumerate(fits_nll):
                pdf_vals = normalized_pdf(s_fit, n, y)
                linestyle = '--' if j < 3 else ':'
                label = f"{'NLL' if j < 3 else 'Hist'} Fit {j%3+1}: n={n:.2f}, y={y:.2f}"
                axs[0].plot(s_fit, pdf_vals, linestyle=linestyle, label=label)

            axs[0].set_xlim(0, 0.5)
            axs[0].set_ylim(0, 0.75)
            axs[0].set_title("Zoomed P(s)", fontsize=10)
            try:
                popt, _ = curve_fit(poly_model, centerszoom, densityzoom)
                x_fit = np.linspace(0, 0.5, 300)
                y_fit = poly_model(x_fit, *popt)
                axs[0].plot(x_fit, y_fit, linestyle='--', color='blue', label=fr"$A x^2 + B x^4$" + "\n" + fr"$A={popt[0]:.3f},\ B={popt[1]:.3f}$")
            except RuntimeError:
                print("Fit failed for zoomed-in region")
            # Fit power-law model
            try:
                popt_power, _ = curve_fit(power_law, centerszoom, densityzoom, p0=[1.0, 2.0])
                y_fit_power = power_law(x_fit, *popt_power)
                axs[0].plot(x_fit, y_fit_power, linestyle='--', color='black', label=fr"$A x^{{\beta}}$" + "\n" + fr"$A={popt_power[0]:.2f},\ \beta={popt_power[1]:.2f}$")
            except RuntimeError:
                print("Power-law fit failed")
            overlay_gue_curve(axs[0])
            axs[0].legend(fontsize=6)

            # Full linear
            bins100 = np.linspace(norm_grp.min(), norm_grp.max(), n_bins)
            counts100, edges100 = np.histogram(norm_grp, bins=bins100)
            widths100 = np.diff(edges100)
            centers100 = edges100[:-1] + widths100 / 2
            density100 = counts100 / (len(norm_grp) * widths100)
            axs[1].scatter(centers100, density100, s=4, color=color_cycle[i])
            for j, (n, y) in enumerate(fits_nll):
                pdf_vals = normalized_pdf(s_fit, n, y)
                linestyle = '--' if j < 3 else ':'
                label = f"{'NLL' if j < 3 else 'Hist'} Fit {j%3+1}: n={n:.2f}, y={y:.2f}"
                axs[1].plot(s_fit, pdf_vals, linestyle=linestyle, label=label)
            axs[1].set_title("Full Linear P(s)")
            overlay_gue_curve(axs[1])
            axs[1].legend(fontsize=6)
            axs[1].set_xlim(0.9*centers100.min(), centers100.max()*1.1)
            axs[1].set_ylim(0.9*density100.min(), density100.max()*1.1)

            # Log-linear
            axs[2].semilogy(centers100, density100, marker='.', linestyle='None', markersize=4,
                            color=color_cycle[i])
            for j, (n, y) in enumerate(fits_nll):
                pdf_vals = normalized_pdf(s_fit, n, y)
                linestyle = '--' if j < 3 else ':'
                label = f"{'NLL' if j < 3 else 'Hist'} Fit {j%3+1}: n={n:.2f}, y={y:.2f}"
                axs[2].semilogy(s_fit, pdf_vals, linestyle=linestyle, label=label)
            axs[2].set_title("Log-Linear")
            overlay_gue_curve(axs[2])
            axs[2].legend(fontsize=6)
            axs[2].set_xlim(0.2*centers100.min(), centers100.max()*1.1)
            axs[2].set_ylim(1e-5, 5)

            # Log-log
            pos = norm_grp[norm_grp > 0]
            if len(pos) > 1:
                bins_log = np.logspace(np.log10(pos.min()), np.log10(pos.max()), n_bins)
                counts_log, edges_log = np.histogram(pos, bins=bins_log)
                centers_log = edges_log[:-1] * np.sqrt(edges_log[1:] / edges_log[:-1])
                density_log = counts_log / (len(pos) * np.diff(edges_log))
                axs[3].loglog(centers_log, density_log, marker='.', linestyle='None', markersize=4,
                              color=color_cycle[i])
                for j, (n, y) in enumerate(fits_nll):
                    pdf_vals = normalized_pdf(s_fit, n, y)
                    linestyle = '--' if j < 3 else ':'
                    label = f"{'NLL' if j < 3 else 'Hist'} Fit {j%3+1}: n={n:.2f}, y={y:.2f}"
                    axs[3].loglog(s_fit, pdf_vals, linestyle=linestyle, label=label)
                overlay_gue_curve(axs[3])
                axs[3].set_title("Log-Log")
                axs[3].legend(fontsize=6)
                axs[3].set_xlim(0.2*centers100.min(), centers100.max()*1.1)
                axs[3].set_ylim(1e-5, 2)
                axs[3].set_ylim(1e-5, 5)

            # === Q-Q Plot vs GUE: decimated + percentile quantiles ===

            # ---- (1) Decimated plot: uniform spacing in index space ----
            norm_sorted = np.sort(norm_grp)
            N = len(norm_sorted)
            probs_uniform = (np.arange(1, N + 1) - 0.5) / N
            s_grid = np.linspace(1e-3, max(5, norm_sorted[-1] * 1.1), 2000)
            cdf_gue = gue_cdf(s_grid, *fits_nll[2])  # best NLL fit
            inv_cdf_gue = interp1d(cdf_gue, s_grid, bounds_error=False, fill_value='extrapolate')
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

# Analyze folder using new three-scenario parity processing
def analyze_folder_three_scenarios(folder_path, energy_range=(-0.3, 0.3)):
    files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    total = len(files)
    print(f"Processing {total} files.")
    all_seps = {
        'C=+1 to nearest -1': [],
        'C=-1 to nearest +1': [],
        'C=±1 to nearest opposite parity': []
    }

    for idx, f in enumerate(files, 1):
        if idx % 100 == 0:
            print(f"  Processed {idx}/{total} files.")
        data = np.load(os.path.join(folder_path, f))
        if not np.isclose(data['SumChernNumbers'], 1, atol=1e-5):
            continue
        trial = {'eigs': data['eigsPipi'], 'chern': data['ChernNumbers']}
        res = process_trial_three_scenarios(trial, energy_range)
        for name, seps in res.items():
            if len(seps):
                all_seps[name].append(seps)

    all_seps = {k: np.concatenate(v) for k, v in all_seps.items() if v}
    with PdfPages('three_parity_scenarios_analysis.pdf') as pdf:
        generate_scatter_histograms(all_seps, energy_range, pdf)

analyze_folder("/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/N=1024_Mem/", energy_range=(-0.03, 0.03))
# analyze_folder_three_scenarios("/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/N=1024_Mem/", energy_range=(-0.03, 0.03))

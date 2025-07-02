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

def compute_chi_squared(norm_data, model_pdf_func, b=None, num_bins=50):
    # Histogram the data (you get counts, not densities)
    counts, bin_edges = np.histogram(norm_data, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)
    total_counts = np.sum(counts)

    # Evaluate model PDF over bin centers
    model_pdf = model_pdf_func(bin_centers) if b is None else model_pdf_func(bin_centers, b)

    # Expected counts = PDF * bin width * total number of samples
    expected = model_pdf * bin_widths * total_counts

    # Avoid zero expected counts
    mask = expected > 0
    chi2 = np.sum((counts[mask] - expected[mask])**2 / expected[mask])
    dof = np.sum(mask) - (1 if b is not None else 0)  # subtract 1 for fitted parameter

    return chi2, dof, chi2 / dof

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

        chi2, dof, red_chi2 = compute_chi_squared(norm_all, normalized_pdf, b=b_fit)
        print(f"Chi-squared = {chi2:.2f}, DOF = {dof}, reduced = {red_chi2:.2f}")


        groups = split_into_groups(seps)
        fig, axs = plt.subplots(1, 5, figsize=(24, 4))
        fig.suptitle(
            f"{name}  |  Energy: [{energy_range[0]:.3f}, {energy_range[1]:.3f}]  |  Fitted $b = {b_fit:.3f}$",
            fontsize=12)
        
        all_centers_log = []
        all_counts_log = []

        for i, grp in enumerate(groups):
            norm_grp, _ = normalize_separations(grp)
            n_bins = get_dynamic_bin_count(len(norm_grp))

            # # Linear 99% plot
            # cutoff = np.percentile(norm_grp, 99)
            # data99 = norm_grp[norm_grp <= cutoff]
            # if len(data99) > 1:
            #     bins99 = np.linspace(data99.min(), data99.max(), n_bins)
            #     counts99, edges99 = np.histogram(data99, bins=bins99)
            #     widths99 = np.diff(edges99)
            #     centers99 = edges99[:-1] + widths99 / 2
            #     density99 = counts99 / (len(norm_grp) * widths99)
            #     axs[0].scatter(centers99, density99, s=4, color=color_cycle[i])
            #     axs[0].plot(s_fit, normalized_pdf(s_fit, b_fit), 'r--', label=f"Fit b={b_fit:.2f}")

            # axs[0].set_title(f'Linear (99\%)\nN={len(seps)}, bins={n_bins}', fontsize=10)
            # axs[0].set_xlabel(r"$s/\langle s \rangle$")
            # axs[0].set_ylabel(r"$P(s/\langle s \rangle)$")

            # Linear 0.5 plot
            datazoom = norm_grp[norm_grp <= 0.5]
            if len(datazoom) > 1:
                binszoom = np.linspace(datazoom.min(), datazoom.max(), 50)
                countszoom, edgeszoom = np.histogram(datazoom, bins=binszoom)
                widthszoom = np.diff(edgeszoom)
                centerszoom = edgeszoom[:-1] + widthszoom / 2
                densityzoom = countszoom / (len(norm_grp) * widthszoom)

                # Plot histogram points
                axs[0].scatter(centerszoom, densityzoom, s=4, color=color_cycle[i], label='Data')

                # Fit Ax^2 + Bx^4 model to the histogram data
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

                # Existing fit overlay
                axs[0].plot(s_fit, normalized_pdf(s_fit, b_fit), 'r--', label=f"GUE Fit $b={b_fit:.2f}$")

            overlay_gue_curve(axs[0])
            axs[0].set_title(f'Linear (99\%)\nN={len(seps)}, bins={n_bins}', fontsize=10)
            axs[0].set_xlabel(r"$s/\langle s \rangle$")
            axs[0].set_ylabel(r"$P(s/\langle s \rangle)$")
            axs[0].set_xlim(0, 0.5)
            axs[0].set_ylim(0, 0.75)
            axs[0].legend(fontsize=7)


            # Linear 100% plot
            bins100 = np.linspace(norm_grp.min(), norm_grp.max(), n_bins)
            counts100, edges100 = np.histogram(norm_grp, bins=bins100)
            widths100 = np.diff(edges100)
            centers100 = edges100[:-1] + widths100 / 2
            density100 = counts100 / (len(norm_grp) * widths100)
            axs[1].scatter(centers100, density100, s=4, color=color_cycle[i])
            axs[1].plot(s_fit, normalized_pdf(s_fit, b_fit), 'r--',label=f'GUE-Fit: $b={b_fit:.2f}$')
            

            axs[1].set_title(f'Linear (100\%)\nN={len(seps)}, bins={n_bins}', fontsize=10)
            axs[1].set_xlabel(r"$s/\langle s \rangle$")
            axs[1].set_ylabel(r"$P(s/\langle s \rangle)$")
            overlay_gue_curve(axs[1])
            axs[1].legend(fontsize=6)            
            # axs[1].set_xlim(0, 4)

            # Log-linear
            axs[2].semilogy(centers100, density100, marker='.', linestyle='None', markersize=4,
                            color=color_cycle[i], label=f'G{i+1}: {len(norm_grp)} pts')
            axs[2].plot(s_fit, normalized_pdf(s_fit, b_fit), 'r--',label=fr'GUE-Fit: $b={b_fit:.2f}$')
            axs[2].set_title(f"Log-Linear\nN={len(seps)}, bins={n_bins}", fontsize=10)
            axs[2].set_xlabel(r"$s/\langle s \rangle$")
            axs[2].set_ylabel(r"$\log P(s/\langle s \rangle)$")
            overlay_gue_curve(axs[2])
            axs[2].legend(fontsize=6)   
            axs[2].set_ylim(1e-4, 1.7)

            x = np.linspace(norm_grp.min(), norm_grp.max(), 1000)
            kde_full = stats.gaussian_kde(norm_grp)
            axs[2].plot(x, kde_full(x), color="green", label="KDE-Fit", alpha=0.8)
            # axs[2].set_xlim(0, 4)

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
                axs[3].plot(s_fit, normalized_pdf(s_fit, b_fit), 'r--',label=fr'GUE-Fit: $b={b_fit:.2f}$')
                overlay_gue_curve(axs[3])
                axs[3].legend(fontsize=6)

                all_centers_log.extend(centers100)
                all_counts_log.extend(density100)

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

        # # === Fit a power law up to x=0.5 and overlay it === (prev. incorrect!)
        x_thresh = 0.5
        all_centers_log = np.array(all_centers_log)
        all_counts_log = np.array(all_counts_log)
        mask = (all_centers_log > 0) & (all_counts_log > 0) & (all_centers_log <= x_thresh)

        if np.sum(mask) > 2:
            log_x = np.log10(all_centers_log[mask])
            log_y = np.log10(all_counts_log[mask])
            slope, intercept, _, _, _ = stats.linregress(log_x, log_y)

            x_fit = np.logspace(np.log10(min(all_centers_log)), np.log10(max(all_centers_log)), 500)
            y_fit = 10**intercept * x_fit**slope
            axs[3].loglog(x_fit, y_fit, linestyle='--', color='black', label=fr'$x^{{{slope:.2f}}}$')
            axs[3].axvline(x=x_thresh, color='gray', linestyle=':', linewidth=1)
            axs[3].legend(fontsize=6)

        # === Fit a power law in linear space up to x=0.5 === linear version, worse
        # x_thresh = 0.5
        # all_centers_log = np.array(all_centers_log)
        # all_counts_log = np.array(all_counts_log)

        # mask = (all_centers_log > 0) & (all_counts_log > 0) & (all_centers_log <= x_thresh)
        # x_data = all_centers_log[mask]
        # y_data = all_counts_log[mask]

        # if len(x_data) > 2:
        #     try:
        #         popt, _ = curve_fit(power_law, x_data, y_data, p0=[1.0, 2.0])
        #         A_fit, beta_fit = popt
        #         x_fit = np.linspace(x_data.min(), x_data.max(), 500)
        #         y_fit = power_law(x_fit, A_fit, beta_fit)
        #         axs[3].plot(x_fit, y_fit, linestyle='--', color='black', label=fr"$A x^{{\beta}}$" + "\n" + fr"$A={A_fit:.2f},\ \beta={beta_fit:.2f}$")
        #         axs[3].axvline(x=x_thresh, color='gray', linestyle=':', linewidth=1)
        #         axs[3].legend(fontsize=6)
        #     except RuntimeError:
        #         print("Power-law fit failed in linear space")

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

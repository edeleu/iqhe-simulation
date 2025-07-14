import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os
from tqdm import tqdm
from matplotlib import rc
from scipy.integrate import quad, cumulative_trapezoid as cumtrapz
from scipy.stats import kstest
from scipy.interpolate import interp1d

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
    "figure.figsize": (3.4, 3),          # Default figure size for single-column plots
    "lines.linewidth": 1,
    "grid.alpha": 0.3,
    "axes.grid": True
})

# Configure the font rendering with LaTeX for compatibility
rc('text.latex', preamble=r'\usepackage{amsmath}')  # Allows using AMS math symbols

## FIRST STEP: PROCESS AND COMPUTE R-VALUES DEPENDING ON DESIRED SCHEME
def local_r_values(sorted_eigs):
    """
    Given sorted eigenvalues: E_0 < E_1 < ... < E_{M-1},
    compute local r_i for i=1..(M-2):
        r_i = min( E_{i+1}-E_i, E_i - E_{i-1} ) / max(...)
    Return (E_mid, r_vals), where E_mid[i] = E_i for i=1..(M-2).
    """
    M = len(sorted_eigs)
    if M < 3:
        return np.array([]), np.array([])

    diffs = np.diff(sorted_eigs)
    E_mid = sorted_eigs[1:-1]
    r_list = []

    for i in range(1, M-1):
        s_left  = diffs[i-1]
        s_right = diffs[i]
        r_i = min(s_left, s_right) / max(s_left, s_right)
        r_list.append(r_i)

    return E_mid, np.array(r_list)

def r_overlapping(sorted_eigs):
    """
    Computes r_overlapping^(n) = (E_{i+2} - E_i) / (E_{i+1} - E_{i-1})
    for i = 1 to M-3. Returns E_center = E_i, and r_values.
    """
    M = len(sorted_eigs)
    if M < 4:
        return np.array([]), np.array([])

    E_center = sorted_eigs[1:M-2]
    numerator = sorted_eigs[3:M] - sorted_eigs[1:M-2]
    denominator = sorted_eigs[2:M-1] - sorted_eigs[0:M-3]
    r_values = numerator / denominator
    return E_center, r_values

def r_nonoverlapping(sorted_eigs):
    """
    Computes r_nonoverlapping^(2) = (E_{i+4} - E_{i+2}) / (E_{i+2} - E_i)
    for i = 0 to M-5. Returns E_center = E_{i+2}, and r_values.
    """
    M = len(sorted_eigs)
    if M < 5:
        return np.array([]), np.array([])

    E_center = sorted_eigs[2:M-2]
    r_values = (sorted_eigs[4:M] - sorted_eigs[2:M-2]) / (sorted_eigs[2:M-2] - sorted_eigs[0:M-4])
    return E_center, r_values

def filter_chern(eigs, cvals, cfilter):
    """
    If cfilter is None => return all,
    else return only E where cvals in cfilter.
    """
    if cfilter is None or cvals is None:
        return eigs
    if isinstance(cfilter, int):
        cfilter = [cfilter]
    mask = np.isin(cvals, cfilter)
    return eigs[mask]

###############################################################################
# 1a) Gather all (E, r) for a subdirectory
###############################################################################

def gather_local_r_hexbin(subdir_path, cfilter=None, sign_flip=True):
    """
    For each .npz in subdir_path:
      - load eigenvalues, filter by cfilter if desired
      - sort, compute local r-values => (E_mid, r)
      - optionally do the same for -E (sign_flip=True)
    Concatenate all results for that subdirectory.

    Returns E_all, r_all arrays.
    """
    file_list = [
        os.path.join(subdir_path, f)
        for f in os.listdir(subdir_path)
        if f.endswith('.npz')
    ]
    if not file_list:
        return np.array([]), np.array([])

    E_accum = []
    r_accum = []

    for fpath in tqdm(file_list, desc=f"Loading {subdir_path}"):
        data = np.load(fpath)
        eigs = data['eigsPipi']
        cvals = data.get('ChernNumbers', None)

        # Filter by cfilter
        E_filtered = filter_chern(eigs, cvals, cfilter)
        if len(E_filtered) < 3:
            continue
        E_sorted = np.sort(E_filtered)
        E_mid_pos, r_pos = r_nonoverlapping(E_sorted)
        E_accum.append(E_mid_pos)
        r_accum.append(r_pos)

        if sign_flip:
            # Treat negative energies as separate "trial"
            E_neg = -E_sorted
            E_neg_sorted = np.sort(E_neg)
            E_mid_neg, r_neg = r_nonoverlapping(E_neg_sorted)
            E_accum.append(E_mid_neg)
            r_accum.append(r_neg)

    if not E_accum:
        return np.array([]), np.array([])

    return np.concatenate(E_accum), np.concatenate(r_accum)


# === Define folded Wigner-Dyson PDF ===
def wigner_dyson_pdf(r, beta):
    Z = compute_normalization_constant(beta)
    num = (r + r**2)**beta
    denom = (1 + r + r**2)**(1 + 1.5 * beta)
    return num / (Z * denom)

# === Compute normalization constant for folded PDF ===
def compute_normalization_constant(beta):
    integrand = lambda r: (r + r**2)**beta / (1 + r + r**2)**(1 + 1.5 * beta)
    val, _ = quad(integrand, 0, np.inf, limit=100)
    return val

def empirical_cdf(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    yvals = np.arange(1, n+1) / n
    return sorted_data, yvals

def model_cdf(r_grid, beta):
    pdf_vals = wigner_dyson_pdf(r_grid, beta)
    cdf_vals = cumtrapz(pdf_vals, r_grid, initial=0)
    return cdf_vals / cdf_vals[-1]

def perform_ks_test(r_vals, beta):
    r_grid = np.linspace(1e-4, np.percentile(r_vals, 99.5)*1.5, 2000)
    model_cdf_vals = model_cdf(r_grid, beta)
    model_interp = interp1d(r_grid, model_cdf_vals, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))
    return kstest(r_vals, lambda x: model_interp(np.asarray(x)))


def qq_plot(ax, data, beta):
    data = np.sort(np.asarray(data))
    n = len(data)
    r_grid = np.linspace(1e-4, np.percentile(data, 99.5)*1.5, 2000)  # extended domain

    cdf_vals = model_cdf(r_grid, beta)
    inv_cdf_interp = interp1d(cdf_vals, r_grid, kind='linear', bounds_error=False, fill_value="extrapolate")

    probs_uniform = (np.arange(1, n + 1) - 0.5) / n
    th_quantiles_uniform = inv_cdf_interp(probs_uniform)
    step = max(1, n // 10000)
    ax.plot(th_quantiles_uniform[::step], data[::step], '.', markersize=3, alpha=0.6, label='Sampled 10k')

    quantile_probs = np.linspace(0.01, 0.99, 200)
    emp_q = np.quantile(data, quantile_probs)
    th_q = inv_cdf_interp(quantile_probs)
    ax.plot(th_q, emp_q, 'o', markersize=2.5, alpha=0.9, label='200 Quantiles')

    lims = [0, max(th_q.max(), emp_q.max(), data.max())]
    ax.plot(lims, lims, 'k--', lw=0.8, label='Ideal')

    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Empirical Quantiles')
    ax.set_title('Q-Q Plot')
    ax.legend(fontsize=6, loc='upper left', frameon=False)
    ax.grid(True, alpha=0.3)

# === Fitting function: negative log-likelihood ===
def neg_log_likelihood(beta, data):
    pdf_vals = wigner_dyson_pdf(data, beta)
    if np.any(pdf_vals <= 0):
        return np.inf
    return -np.sum(np.log(pdf_vals))


def main_fitr(folder_path, overlay_n_values,energy_window):
    subdirs = [
        d for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d))
    ]

    for subdir in subdirs:
        try:
            n_value = subdir.split('=')[-1].split('_')[0]
            n_value = int(n_value)
        except:
            continue

        if n_value not in overlay_n_values:
            continue

        subdir_path = os.path.join(folder_path, subdir)
        print(f"=== Subdir: {subdir}, N={n_value} ===")

        Evals, rvals = gather_local_r_hexbin(subdir_path, sign_flip=True)
        mask = (Evals >= energy_window[0]) & (Evals <= energy_window[1])
        r_values = rvals[mask]

        # === Fit beta ===
        res = minimize_scalar(lambda b: neg_log_likelihood(b, r_values), bounds=(0.1, 12), method='bounded')
        beta_fit = res.x

        # === Plotting ===
        bins = np.linspace(0, 1, 50)
        hist, bin_edges = np.histogram(r_values, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        cutoff = np.percentile(r_values, 99.5)
        data99 = r_values[r_values <= cutoff]
        if len(data99) > 1:
            bins99 = np.linspace(data99.min(), data99.max(), 50)
            density99, bin_edges = np.histogram(data99, bins=bins99)
            widths99 = np.diff(bin_edges)
            bin_centers = bin_edges[:-1] + widths99 / 2
            hist = density99 / (len(r_values) * widths99)
            # axs[0,0].scatter(bin_centers, density99, s=4)

        pdf_fit = wigner_dyson_pdf(bin_centers, beta_fit)

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        r_mean = np.mean(r_values)

        # Linear plot
        axs[0, 0].plot(bin_centers, hist, 'o', label='Data')
        axs[0, 0].plot(bin_centers, pdf_fit, label=f'Fit ($\\beta={beta_fit:.2f}$)')
        axs[0, 0].set_title("Linear Scale")
        axs[0,0].axvline(r_mean, color='red', linestyle=':', linewidth=1.5,
               label=fr"$\langle r \rangle = {r_mean:.3f}$")
        axs[0, 0].legend()
        axs[0, 0].set_xlabel(r"$r$")
        axs[0, 0].set_ylabel(r"$P(r)$")

        # Semilog-y
        axs[0, 1].semilogy(bin_centers, hist, 'o')
        axs[0, 1].semilogy(bin_centers, pdf_fit)
        axs[0,1].axvline(r_mean, color='red', linestyle=':', linewidth=1.5,
        label=fr"$\langle r \rangle = {r_mean:.3f}$")

        axs[0, 1].set_title("Semilog-y")
        axs[0, 1].set_xlabel(r"$r$")
        axs[0, 1].set_ylabel(r"$log P(r)$")

        # Log-log
        axs[1, 0].loglog(bin_centers, hist, 'o')
        axs[1, 0].loglog(bin_centers, pdf_fit)
        axs[1,0].axvline(r_mean, color='red', linestyle=':', linewidth=1.5,
            label=fr"$\langle r \rangle = {r_mean:.3f}$")
        axs[1, 0].set_title("Log-Log")
        axs[1, 0].set_xlabel(r"log $r$")
        axs[1, 0].set_ylabel(r"$log P(r)$")

        # Q-Q plot
        # probplot(r_values, dist=lambda b: folded_pdf(b, beta_fit), plot=axs[1, 1])
        # axs[1, 1].set_title("Q-Q Plot")
        qq_plot(axs[1, 1], r_values, beta_fit)

        # === Report KS statistic ===
        ks_stat, p_val = perform_ks_test(r_values, beta_fit)

        # Add annotation
        axs[0, 0].text(0.97, 0.97, f"KS = {ks_stat:.4f}\np = {p_val:.4g}",
                    ha='right', va='top', transform=axs[0, 0].transAxes,
                    fontsize=11, bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

        plt.tight_layout()
        plt.show()

        print(f"Fitted beta: {beta_fit:.4f}")
        print(f"KS statistic: {ks_stat:.4f}")
        print(f"KS p-value: {p_val:.4g}")

        fig.tight_layout()
        fig.savefig(f"r_fitted_distribution{n_value}.pdf")

###############################################################################
# Example usage
###############################################################################
if __name__ == "__main__":
    folder_path = "/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/"  # adjust this path as needed
    overlay_n_values = [64,128,256,512,1024,2048]  # just an example
    overlay_n_values = [1024]

    main_fitr(folder_path, overlay_n_values,energy_window=[-0.03,0.03])
    # print(compute_normalization_constant(2))
    # print()
    # print((4/81) * np.pi / np.sqrt(3))
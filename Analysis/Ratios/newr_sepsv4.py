#NO BINS
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy

# If you want to do LOWESS, you'll need statsmodels:
# import statsmodels.api as sm
# from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib import rc

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

def generalized_r(r, b=2):
    """
    PDF of the general r-distribution
    where b is the Wigner beta value.
    """
    normconstant = (4/81) * np.pi / np.sqrt(3)
    numerator = (r + r**2)**b
    denominator = (1 + r + r**2)**(1 + 3*b/2)
    return numerator / (normconstant * denominator)

def generalized_folded_r(r, b=2):
    """
    folded, normalized PDF of the general r-distribution
    r here is r_tilda (min-max) from 0 --> 1
    """

    return generalized_r(r,b) + (1/(r**2))*generalized_r(1/r,b)

def overlay_gue_curve(ax, num_points=1000, label="Reference GUE", color="green", linestyle="--"):
    r_vals = np.linspace(0.001, 1, num_points)
    ax.plot(r_vals, generalized_folded_r(r_vals), label=fr'Folded GUE ($\beta=2$)',color=color)

###############################################################################
# 1) Local r-values
###############################################################################

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
# 2) Gather all (E, r) for a subdirectory
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

###############################################################################
# 3) Plot: 2D Hexbin + LOWESS
###############################################################################
def compute_quantile_bin_edges(E, tail_frac=0.0025, n_center=35):
    """
    Generate bin edges for quantile-based binning with specified tail fractions.

    This creates:
      - One left tail bin from min(E) to q_min (quantile at tail_frac)
      - n_center bins between q_min and q_max (evenly spaced in quantile space)
      - One right tail bin from q_max to max(E)

    Parameters
    ----------
    E : array-like
        Input 1D array of energy values.
    tail_frac : float
        Fraction of data to allocate to each tail (e.g., 0.005 for 0.5%).
    n_center : int
        Number of bins in the center region between q_min and q_max.

    Returns
    -------
    bin_edges : np.ndarray
        Array of length (n_center + 3), containing the bin edges:
        [min(E), q_min, ..., q_max, max(E)]
    """
    E = np.asarray(E)
    if E.size < 2:
        return np.array([])

    # Sort once
    E_sorted = np.sort(E)

    # Tail quantiles
    q_min = np.quantile(E_sorted, tail_frac)
    q_max = np.quantile(E_sorted, 1 - tail_frac)

    # Center bin quantile edges
    center_qs = np.linspace(tail_frac, 1 - tail_frac, n_center + 1)
    center_edges = np.quantile(E_sorted, center_qs)

    # Combine edges: min(E) → q_min → center bins → q_max → max(E)
    bin_edges = [E_sorted[0], q_min]
    bin_edges.extend(center_edges[1:-1])  # exclude q_min and q_max to avoid duplication
    bin_edges.append(q_max)
    bin_edges.append(E_sorted[-1])

    return np.array(bin_edges)

def plot_hexbin_lowess(E, r, ax, gridsize=100, frac=0.05):
    """
    Plot a 2D hexbin of the points (E, r) on the given Axes 'ax',
    then overlay a LOWESS curve of r vs. E.

    Parameters
    ----------
    E, r : 1D arrays
    ax   : matplotlib Axes
    gridsize : int, hexbin resolution
    frac : float, fraction for LOWESS smoothing
    """
    # 2D hexbin
    hb = ax.hexbin(E, r, gridsize=gridsize,
                   extent=[E.min(), E.max(), r.min(), r.max()],
                   cmap="viridis")
    # colorbar
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("counts")

    # Sort E, r for LOWESS
    idx_sort = np.argsort(E)
    E_sorted = E[idx_sort]
    r_sorted = r[idx_sort]

    # Run LOWESS
    smoothed = sm.nonparametric.lowess(endog=r_sorted, exog=E_sorted, frac=frac,is_sorted=True,delta=0.001)
    # smoothed is shape (N,2): [ [E0, r_smooth0], [E1, r_smooth1], ... ]

    ax.plot(smoothed[:,0], smoothed[:,1], 'r-', linewidth=2,
            label=f"LOWESS frac={frac}")

    ax.set_xlim(E.min(), E.max())
    ax.set_ylim(r.min(), r.max())
    # ax.legend()


def plot_hexbin_with_fits(E, r, ax, gridsize=100, frac=0.05, bins=250):
    """
    Plot a 2D hexbin of (E, r), and overlay:
      - LOWESS regression (red)
      - Rolling average (white dashed)
      - Rolling median (cyan solid)

    Parameters
    ----------
    E, r      : 1D arrays of equal length
    ax        : matplotlib Axes to plot on
    gridsize  : int, number of hex cells in x direction
    frac      : LOWESS smoothing parameter
    bins      : number of bins for rolling avg / median
    """

    # Step 1: 2D Hexbin plot
    hb = ax.hexbin(E, r, gridsize=gridsize,
                   extent=[E.min(), E.max(), r.min(), r.max()],
                   cmap="viridis")
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("counts")

    # Step 2: LOWESS Fit (fast variant)
    idx = np.argsort(E)
    E_sorted = E[idx]
    r_sorted = r[idx]
    delta = 0.005 * (E.max() - E.min())  # adjust if needed

    smoothed = sm.nonparametric.lowess(r_sorted, E_sorted,
                      frac=frac,
                      delta=delta,
                      is_sorted=True,
                      return_sorted=True)
    ax.plot(smoothed[:, 0], smoothed[:, 1], 'r-', lw=2, label=f"LOWESS frac={frac}")

    # Step 3: Rolling avg and median (in E-space)
    bin_edges = np.linspace(E.min(), E.max(), bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    digitized = np.digitize(E, bin_edges) - 1

    r_mean = np.full(bins, np.nan)
    r_median = np.full(bins, np.nan)
    for i in range(bins):
        in_bin = digitized == i
        if np.any(in_bin):
            r_mean[i] = np.mean(r[in_bin])
            r_median[i] = np.median(r[in_bin])

    # Step 4: Overlay other fits
    ax.plot(bin_centers, r_mean, 'w--', lw=1.5, label="Rolling Avg")
    ax.plot(bin_centers, r_median, 'c-', lw=1.5, label="Rolling Median")

    # Step 5: Final cleanup
    ax.set_xlim(E.min(), E.max())
    ax.set_ylim(r.min(), r.max())
    ax.set_xlabel("Energy E")
    ax.set_ylabel("Local r")
    # ax.legend(loc="upper right")

def plot_hexbin_with_quantile_fits(E, r, ax, frac=0.05, gridsize=100,
                                   tail_frac=0.0025, n_center=35):
    """
    Plot 2D hexbin of (E, r) and overlay:
      - LOWESS (red)
      - Rolling average (white dashed)
      - Rolling median (cyan solid)
    using **quantile-based binning** for the rolling statistics.

    Parameters
    ----------
    E, r        : 1D arrays
    ax          : matplotlib Axes
    frac        : LOWESS smoothing fraction
    gridsize    : hexbin resolution
    tail_frac   : tail quantile fraction (e.g. 0.005 for 0.5%)
    n_center    : number of central bins
    """

    # 1. Hexbin
    # hb = ax.hexbin(E, r, gridsize=gridsize,
    #                extent=[E.min(), E.max(), r.min(), r.max()],
    #                cmap="viridis")

    # hb = ax.hexbin(E, r, gridsize=gridsize,
    #                extent=[E.min(), E.max(), 0.75, 1.25],
    #                cmap="viridis")
    # cb = ax.figure.colorbar(hb, ax=ax)
    # cb.set_label("counts")

    # 2. LOWESS
    # idx = np.argsort(E)
    # E_sorted = E[idx]
    # r_sorted = r[idx]
    # delta = 0.01 * (E.max() - E.min())
    # smoothed = lowess(r_sorted, E_sorted,
    #                   frac=frac,
    #                   delta=delta,
    #                   is_sorted=True,
    #                   return_sorted=True)
    # ax.plot(smoothed[:, 0], smoothed[:, 1], 'r-', lw=2, label=f"LOWESS") # frac={frac}

    # 3. Quantile-based binning for average and median
    bin_edges = compute_quantile_bin_edges(E, tail_frac=tail_frac, n_center=n_center)

    ## ALTERNATIVE FIXED-BIN SIZE METHOD
    bin_width = 0.05
    E_min = E.min()
    E_max = E.max()

    # Extend bounds to align symmetrically around 0
    E_left = np.floor((E_min + 0.5 * bin_width) / bin_width) * bin_width
    E_right = np.ceil((E_max - 0.5 * bin_width) / bin_width) * bin_width

    # Create bins from left to right centered on 0
    bin_centers = np.arange(E_left, E_right + bin_width, bin_width)
    bin_edges = bin_centers - 0.5 * bin_width
    bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)  # one extra for last right edge

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    digit = np.digitize(E, bin_edges) - 1

    r_avg = np.full(len(bin_centers), np.nan)
    r_trimavg = np.full(len(bin_centers), np.nan)
    r_med = np.full(len(bin_centers), np.nan)
    r_sem = np.full(len(bin_centers), np.nan)

    for i in range(len(bin_centers)):
        mask = digit == i
        if np.any(mask):
            r_avg[i] = np.mean(r[mask])
            r_trimavg[i] = scipy.stats.trim_mean(r[mask],0.25)
            r_med[i] = np.median(r[mask])
            r_sem[i] = np.std(r[mask], ddof=1) / np.sqrt(len(r[mask]))

    # 4. Overlay trend lines
    # ax.plot(bin_centers, r_avg, color='white', linestyle='--', marker='o', lw=1.5,markersize=2, label="Avg")
    # ax.plot(bin_centers, r_med, color='cyan',  markersize=2, marker='D', lw=1.5, label="Median")
    # ax.scatter(bin_centers, r_avg, color='white', s=12, marker='o', label="Avg points", zorder=3)
    # ax.scatter(bin_centers, r_med, color='cyan',  s=12, marker='D', label="Median points", zorder=3)
    ax.plot(bin_centers, r_avg, color='gray', marker='.', lw=1.5, label="Average")
    ax.plot(bin_centers, r_med, color='orange', marker='.', lw=1.5, label="Median")
    ax.plot(bin_centers, r_trimavg, color='blue', marker='.', lw=1.5, label="Trimmed Mean")

    # Find index of center bin
    center_idx = np.argmin(np.abs(bin_centers))

    # Retrieve precomputed stats
    r_mean = r_avg[center_idx]
    r_trimmed = r_trimavg[center_idx]
    r_median = r_med[center_idx]

    # Format label text
    text_str = '\n'.join([
        fr"$\langle r \rangle = {r_mean:.3f}$",
        fr"Median $r = {r_median:.3f}$",
        fr"Trimmed $r = {r_trimmed:.3f}$"
    ])

    # Plot label in upper left of axis
    ax.text(0.02, 0.98, text_str,
            transform=ax.transAxes,
            va='bottom', ha='right',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))


    # ax.errorbar(bin_centers, r_avg, yerr=r_sem,
    #             fmt='o-', color='white', lw=1.5,
    #             ecolor='gray', elinewidth=1.0, capsize=2.5, markersize=2,
    #             label="Avg ± SE", zorder=3)

    # 5. Format plot
    ax.set_xlim(E.min(), E.max())
    ax.set_ylim(r.min(), r.max())

    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(0.9, 1.5)
    ax.set_xlabel("Energy E")
    ax.set_ylabel("Local r")
    # ax.legend(loc="upper right")

def plot_pr_histogram(rvals, ax, label, bins=75):
    """
    Plot a normalized histogram of rvals on given Axes,
    with a vertical dotted line showing the average.
    """
    # Histogram
    ax.hist(rvals, bins=bins, density=True,
            alpha=0.8, color='skyblue', edgecolor=None)

    # Mean value
    r_mean = np.mean(rvals)
    ax.axvline(r_mean, color='red', linestyle=':', linewidth=1.5,
               label=fr"$\langle r \rangle = {r_mean:.3f}$")

    # Axis setup
    overlay_gue_curve(ax)
    ax.set_xlim(0, np.max(rvals) * 1.1)
    ax.set_ylim(0)
    ax.grid(True, alpha=0.3)
    ax.set_title(label)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$P(r)$")
    ax.legend(loc="upper right", fontsize=7)


###############################################################################
# 4) Main function: 2x2 subplots for four Chern filters
###############################################################################

def main_hexbin_lowess(folder_path, overlay_n_values, frac=0.05, gridsize=100):
    """
    For each subdir named e.g. "N=128_...", if 128 is in overlay_n_values,
    gather local r-values, plot them as a hexbin + LOWESS for 4 filters:
      - All Chern
      - C=+1
      - C=-1
      - C=0
    We do a 2x2 figure. Then show or save it.
    """
    subdirs = [
        d for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d))
    ]
    cfilters = [
        ("All Chern",  None),
        ("C=0",  0),
        ("C=+1", 1),
        ("C=-1", -1),
        # ("|C|=1", [-1,1]),
    ]

    for subdir in subdirs:
        # e.g. "N=128_..."
        try:
            n_value = subdir.split('=')[-1].split('_')[0]
            n_value = int(n_value)
        except:
            continue

        if n_value not in overlay_n_values:
            continue

        subdir_path = os.path.join(folder_path, subdir)
        print(f"=== Subdir: {subdir}, N={n_value} ===")

        fig, axes = plt.subplots(2, 2, figsize=(6.8,6))
        axes = axes.ravel()

        ####### Histogram Plot
        fig2, axes2 = plt.subplots(2, 2, figsize=(6.8, 6))
        axes2 = axes2.ravel()
        # fig.suptitle(f"N={n_value}: 2D Hexbin + LOWESS (local r)", fontsize=14)

        for ax_idx, (label, cfilt) in enumerate(cfilters):
            ax = axes[ax_idx]
            Evals, rvals = gather_local_r_hexbin(subdir_path, cfilter=cfilt, sign_flip=True)
            if len(Evals) == 0:
                ax.text(0.5, 0.5, f"No data for {label}",
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(label)
                continue

            # plot_hexbin_lowess(Evals, rvals, ax, gridsize=gridsize, frac=frac)
            plot_hexbin_with_quantile_fits(Evals, rvals, ax, gridsize=gridsize, frac=frac)

            ax.set_title(label)
            # ax.set_xlabel("E")
            ax.set_xlabel(r"$E$")
            ax.set_ylabel(r"Separation Ratio, $r$")
            # ax.axhline(0.3863, color='red', linestyle='--', label='Poisson Value 0.3863')
            # ax.axhline(0.60266, color='green', linestyle='--', label='GUE Value 0.60266')
            ax.axhline(1.0980, color='green', linestyle='--', label='GUE Value 1.0980')

            # Plot histogram
            ax2 = axes2[ax_idx]
            plot_pr_histogram(rvals, ax2, label)
            ax2.set_title(label)
            ax2.set_xlabel(r"$r$")
            ax2.set_ylabel(r"$P(r)$")


        fig.tight_layout()
        fig2.tight_layout()

        # plt.show()  # or 
        fig.savefig(f"hexbin_lowess_N{n_value}vQuantNewV3.pdf")
        fig2.savefig(f"histogram_Pr_N{n_value}.pdf")

###############################################################################
# Example usage
###############################################################################
if __name__ == "__main__":
    folder_path = "/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/"  # adjust this path as needed
    overlay_n_values = [64,128,256,512,1024,2048]  # just an example
    overlay_n_values = [1024]

    main_hexbin_lowess(folder_path, overlay_n_values,
                       frac=0.05, gridsize=100)

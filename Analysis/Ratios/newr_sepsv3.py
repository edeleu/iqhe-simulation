#local r method
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
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
###############################################################################
# Utility functions
###############################################################################

def local_r_values(sorted_eigs):
    """
    Given a sorted array of eigenvalues: E_0 < E_1 < ... < E_{M-1},
    compute r_i for i = 1..(M-2), where
       r_i = min( E_{i+1}-E_i, E_i - E_{i-1} ) / max(...)
    Return (E_mid, r_values), where E_mid[i] = E_i for i in [1..M-2],
    and r_values[i] is the corresponding local r.
    """
    M = len(sorted_eigs)
    if M < 3:
        return np.array([]), np.array([])  # not enough points

    diffs = np.diff(sorted_eigs)  # length = M-1 => diffs[i] = E_{i+1} - E_i
    E_mid = sorted_eigs[1:-1]     # We'll associate r_i with the "middle" E_i, i=1..M-2
    r_vals = []

    for i in range(1, M-1):
        s_left  = diffs[i-1]  # E_i - E_{i-1}
        s_right = diffs[i]    # E_{i+1} - E_i
        r_i = min(s_left, s_right) / max(s_left, s_right)
        r_vals.append(r_i)

    return E_mid, np.array(r_vals)

# def compute_quantile_bin_edges(all_energies, bin_frac=0.05):
#     """
#     Unchanged from before: gather all_energies, sort them,
#     create n_bins = 1/bin_frac bins. Return bin_edges.
#     """
#     if len(all_energies) < 2:
#         return np.array([])

#     sortedE = np.sort(all_energies)
#     n_bins = int(round(1.0 / bin_frac))  # e.g. bin_frac=0.05 => 20 bins
#     if n_bins < 1:
#         n_bins = 1

#     qs = np.linspace(0, 1, n_bins + 1)
#     bin_edges = np.quantile(sortedE, qs)
#     return bin_edges

def compute_chunk_bin_edges(all_energies, chunk_size=10000):
    """
    Splits 'all_energies' into consecutive chunks of size ~chunk_size in sorted order,
    and returns the bin edges as [min_value, ..., max_value].
    
    For example, if chunk_size=10000:
      - The first bin goes from sortedE[0] to sortedE[9999] (inclusive),
      - The second bin goes from sortedE[9999] to sortedE[19999],
      - ... and so on,
      - The final bin ends at sortedE[-1].
    
    This is analogous to the original quantile-based function, except instead of
    ensuring each bin has the same fraction of data, we ensure each bin has a
    fixed number (~chunk_size) of points.

    Returns
    -------
    bin_edges : np.ndarray
        An array of length (num_bins + 1),
        where num_bins = ceil(N / chunk_size).
        The first edge is min(all_energies), the last edge is max(all_energies).
        This matches how the original 'compute_quantile_bin_edges' starts at the
        minimum and ends at the maximum, without using +/-∞.
    """
    energies = np.asarray(all_energies)
    n = len(energies)
    if n < 2:
        # Not enough data
        return np.array([])

    sortedE = np.sort(energies)
    
    # If we have fewer points than chunk_size, everything is in one bin:
    if n <= chunk_size:
        # Single bin covering [lowest_value .. highest_value]
        return np.array([sortedE[0], sortedE[-1]])

    # Number of bins:
    num_bins = int(np.ceil(n / float(chunk_size)))

    bin_edges = []
    # 1) Start at the minimum value
    bin_edges.append(sortedE[0])
    
    # 2) Interior chunk boundaries
    for i in range(1, num_bins):
        idx = i * chunk_size - 1
        if idx >= n - 1:
            break
        bin_edges.append(sortedE[idx])
    
    # 3) End at the maximum value
    bin_edges.append(sortedE[-1])
    
    return np.array(bin_edges)

def compute_quantile_bin_edges(E, tail_frac=0.0025, n_center=25):
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

def filter_chern(eigs, cvals, chern_filter):
    """
    If chern_filter is None => return all.
    If cvals is None => return all.
    Otherwise, filter by cvals in chern_filter.
    """
    if chern_filter is None or cvals is None:
        return eigs
    if isinstance(chern_filter, int):
        chern_filter = [chern_filter]
    mask = np.isin(cvals, chern_filter)
    return eigs[mask]

###############################################################################
# Core function that implements the "local r-value" approach with binning
###############################################################################

def analyze_subdir_local_r(subdir_path, bin_frac=0.05, chern_filter=None):
    """
    1. Gather all .npz in subdir.
    2. For each file:
       - filter eigenvalues by chern_filter
       - sort them
       - compute local r-values for interior points
         => E_mid[i], r[i]
       - do the same for negative ( sign-flipped ) as a separate trial
    3. We'll store (E_mid, r) from all trials in a big list.
    4. Create one unified set of bin edges from *all E_mid from all trials* (pos+neg).
    5. Assign each r to a bin based on E_mid.
    6. Average across all r in each bin => mean, also compute stderror => std/sqrt(N).
    7. Return bin_centers, avg_rvals, stderrs, etc.
    """
    # 1) Gather file_list
    file_list = [os.path.join(subdir_path, f)
                 for f in os.listdir(subdir_path)
                 if f.endswith('.npz')]
    if not file_list:
        return None
    
    all_E_for_bins = []   # we'll gather E_mid from pos+neg for all files
    all_r_for_bins = []   # parallel to all_E_for_bins

    # We'll keep them separate in memory, then unify the bin edges afterwards
    E_mid_all = []
    r_vals_all = []
    
    for fpath in file_list:
        data = np.load(fpath)
        cvals = data.get("ChernNumbers", None)
        eigs  = data["eigsPipi"]
        
        # filter
        eigs_filt = filter_chern(eigs, cvals, chern_filter)
        if len(eigs_filt) < 3:
            continue
        
        # Sort
        E_sorted = np.sort(eigs_filt)
        # Local r-values
        E_mid_pos, r_pos = local_r_values(E_sorted)
        
        # Negative
        E_neg = -E_sorted
        E_neg_sorted = np.sort(E_neg)
        E_mid_neg, r_neg = local_r_values(E_neg_sorted)

        # Accumulate them in big arrays
        if len(E_mid_pos) > 0:
            E_mid_all.append(E_mid_pos)
            r_vals_all.append(r_pos)
        if len(E_mid_neg) > 0:
            E_mid_all.append(E_mid_neg)
            r_vals_all.append(r_neg)
    
    if not E_mid_all:
        return None
    
    # Merge into single arrays
    E_mid_all = np.concatenate(E_mid_all)
    r_vals_all = np.concatenate(r_vals_all)
    
    # 2) Build bin edges from E_mid_all bin_frac=bin_frac
    bin_edges = compute_quantile_bin_edges(E_mid_all)
    if len(bin_edges) < 2:
        return None
    
    n_bins = len(bin_edges) - 1

    # 3) We'll accumulate r-values in each bin
    master_bin_rvals = [[] for _ in range(n_bins)]
    
    # For each (E_mid_i, r_i), figure out which bin E_mid_i falls in
    bin_idx = np.searchsorted(bin_edges, E_mid_all, side='right') - 1
    # filter valid indices in [0..n_bins-1]
    valid_mask = (bin_idx >= 0) & (bin_idx < n_bins)
    bin_idx = bin_idx[valid_mask]
    r_vals_valid = r_vals_all[valid_mask]

    # Accumulate
    for i, b in enumerate(bin_idx):
        master_bin_rvals[b].append(r_vals_valid[i])
    
    # 4) Compute average r and std error
    avg_rvals = np.zeros(n_bins, dtype=float)
    std_errs  = np.zeros(n_bins, dtype=float)
    for b in range(n_bins):
        if len(master_bin_rvals[b]) == 0:
            avg_rvals[b] = np.nan
            std_errs[b]  = np.nan
        else:
            arr_b = np.array(master_bin_rvals[b])
            mean_b = np.mean(arr_b)
            std_b  = np.std(arr_b, ddof=1)
            avg_rvals[b] = mean_b
            std_errs[b]  = std_b / np.sqrt(len(arr_b))
    
    # 5) bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    result = {
        'bin_edges':   bin_edges,
        'bin_centers': bin_centers,
        'avg_rvals':   avg_rvals,
        'std_errs':    std_errs
    }
    return result


###############################################################################
# Main: produce per-subdir figures + an overlay figure across N
###############################################################################

def main_local_r(folder_path, overlay_n_values, bin_frac=0.05):
    """
    Similar structure as before:
      - For each subdir that matches N in overlay_n_values, produce one figure w/ 4 subplots
        (All, +1, -1, 0).
      - Also gather results so we can overlay across N in one final figure.
    """
    subdirs = [d for d in os.listdir(folder_path)
               if os.path.isdir(os.path.join(folder_path, d))]
    
    cfilters = [
        ("All Chern",  None),
        ("C=+1", 1),
        ("C=-1", -1),
        ("C=0",  0),
    ]
    # overlay_data[label][N] = results dict
    overlay_data = { label: {} for label, _ in cfilters }
    
    for subdir in subdirs:
        # parse out the N value
        try:
            n_value = subdir.split('=')[-1].split('_')[0]
            n_value = int(n_value)
        except:
            continue
        
        if n_value not in overlay_n_values:
            continue
        
        subdir_path = os.path.join(folder_path, subdir)
        print(f"=== Analyzing subdir: {subdir} (N={n_value}) ===")

        fig, axes = plt.subplots(2, 2, figsize=(6.8, 6))
        axes = axes.ravel()
        fig.suptitle(f"N={n_value} (local r-values), bin_frac={bin_frac}", fontsize=14)
        
        for ax_idx, (label, cval) in enumerate(cfilters):
            ax = axes[ax_idx]
            results = analyze_subdir_local_r(subdir_path,
                                             bin_frac=bin_frac,
                                             chern_filter=cval)
            if results is None:
                ax.set_title(f"{label}: no data")
                ax.set_xlabel("Energy")
                ax.set_ylabel("r")
                continue
            
            x = results['bin_centers']
            y = results['avg_rvals']
            yerr = results['std_errs']
            
            ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=3, label=f"{label}")
            ax.set_title(label)
            ax.set_xlabel("Energy (quantile bins from E_mid)")
            ax.set_ylabel("Average r")
            ax.axhline(0.3863, color='red', linestyle='--', label=None)
            ax.axhline(0.5996, color='green', linestyle='--', label=None)
            
            overlay_data[label][n_value] = results
        
        plt.tight_layout()
        # plt.show()  # or plt.savefig(f"localR_N{n_value}.png")

    # ------------------------------------------------------------------------
    # Now build an overlay figure: 4 subplots, each for a c-filter,
    # overlay lines for each N in overlay_n_values
    # ------------------------------------------------------------------------
    fig_ov, axes_ov = plt.subplots(2, 2, figsize=(6.8,6))
    axes_ov = axes_ov.ravel()
    # fig_ov.suptitle(f"Overlay across N (local r, bin_frac={bin_frac})", fontsize=14)
    
    for ax_idx, (label, _) in enumerate(cfilters):
        ax = axes_ov[ax_idx]
        ax.set_title(label)
        ax.set_xlabel(r"$E$")
        ax.set_ylabel(r"$\langle r \rangle$")
        ax.axhline(0.3863, color='red', linestyle='--', label=None)
        ax.axhline(0.5996, color='green', linestyle='--', label=None)

        for n_val in sorted(overlay_n_values):
            if n_val not in overlay_data[label]:
                continue
            res = overlay_data[label][n_val]
            x = res['bin_centers']
            y = res['avg_rvals']
            yerr = res['std_errs']
            # ax.errorbar(x, y, yerr=yerr, fmt='.-', elinewidth=1.0, capsize=2.5, markersize=2, label=f"N={n_val}")
            ax.plot(x, y, '.-', markersize=1, label=f"N={n_val}")

        ax.legend(frameon=False)
    
    plt.tight_layout()
    # plt.show()  # or 
    plt.savefig("localR_overlayY.pdf")

###############################################################################
# Example usage
###############################################################################
if __name__ == "__main__":
    folder_path = "/Users/eddiedeleu/Desktop/FinalData"  # adjust this path as needed
    overlay_n_values = [64,128,256,512,1024,2048]  # just an example
    bin_frac = 0.01
    main_local_r(folder_path, overlay_n_values, bin_frac)

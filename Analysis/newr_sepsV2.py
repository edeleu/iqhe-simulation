#unified bins overall

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

###############################################################################
# Utility functions
###############################################################################

def calculate_r_values_within_window(eigs):
    """
    Compute r-values from a sorted window of eigenvalues:
      r = min(s_{n}, s_{n+1}) / max(s_{n}, s_{n+1}),
    where s_{n} = E_{n} - E_{n-1}.
    We only compute r for interior points in the sorted array.
    """
    if len(eigs) < 3:
        return np.array([])
    sorted_eigs = np.sort(eigs)
    diffs = np.diff(sorted_eigs)
    r_vals = []
    for i in range(1, len(sorted_eigs) - 1):
        s_prev = diffs[i - 1]
        s_next = diffs[i]
        r_vals.append(min(s_prev, s_next) / max(s_prev, s_next))
    return np.array(r_vals)

def compute_quantile_bin_edges(all_energies, bin_frac=0.05):
    """
    Given a large array of energies from *all* trials (including negative),
    compute a single set of bin edges so each bin contains bin_frac fraction
    of the data. e.g. bin_frac=0.05 => 20 bins.
    Returns bin_edges (array of length n_bins+1).
    """
    if len(all_energies) < 2:
        return np.array([])
    
    sortedE = np.sort(all_energies)
    n_points = len(sortedE)
    n_bins = int(round(1.0 / bin_frac))  # e.g. 0.05 => 20
    if n_bins < 1:
        n_bins = 1
    
    qs = np.linspace(0, 1, n_bins+1)
    bin_edges = np.quantile(sortedE, qs)
    return bin_edges

def filter_chern(eigs, cvals, chern_filter):
    """
    Filter the eigenvalues by cvals == chern_filter.
    If chern_filter=None => no filter => return all.
    If cvals is None => return all.
    """
    if chern_filter is None or cvals is None:
        return eigs
    # If user passed a single integer, convert to list
    if isinstance(chern_filter, int):
        chern_filter = [chern_filter]
    mask = np.isin(cvals, chern_filter)
    return eigs[mask]

def compute_rvals_in_bins(eigs, bin_edges):
    """
    For a given set of bin_edges and a single set of eigenvalues (one trial),
    place them into bins, compute r-values in each bin, and return a dictionary:
      bin_idx -> list of r-values
    """
    if len(eigs) < 3 or len(bin_edges) < 2:
        return {}
    n_bins = len(bin_edges) - 1
    result = {b: [] for b in range(n_bins)}
    
    # Assign each eigenvalue to a bin
    bin_idx = np.searchsorted(bin_edges, eigs, side='right') - 1
    for b in range(n_bins):
        mask_b = (bin_idx == b)
        if np.any(mask_b):
            rvals_b = calculate_r_values_within_window(eigs[mask_b])
            if len(rvals_b) > 0:
                result[b].extend(rvals_b)
    
    return result


###############################################################################
# Core function: analyze all .npz in a given subdir with a specific Chern filter
###############################################################################

def analyze_subdir_unified_bins(subdir_path, bin_frac=0.05, chern_filter=None):
    """
    - Gather *all* eigenvalues from all .npz in subdir (filtered by `chern_filter`).
    - Also gather their negative for bin-edge computation (unify binning).
    - Build a single set of bin_edges from these combined energies.
    - Then for each file, treat E and -E as separate trials. We compute r-values
      in each bin, accumulate them in a global structure.
    - Finally, compute avg_rvals[b], stderror[b] for each bin b, plus bin_centers.
    - Return a dict with the aggregated results.
    """
    # 1) Gather file paths
    file_list = []
    for fname in os.listdir(subdir_path):
        if fname.endswith('.npz'):
            file_list.append(os.path.join(subdir_path, fname))
    if not file_list:
        return None
    
    all_energies_for_bins = []
    # We also store the actual per-file data so we don't reload it
    file_data_list = []
    
    # 2) Collect energies across all files
    for fpath in file_list:
        data = np.load(fpath)
        cvals = data['ChernNumbers'] if 'ChernNumbers' in data else None
        eigs  = data['eigsPipi']
        # Filter by the chosen Chern
        eigs_filt = filter_chern(eigs, cvals, chern_filter)
        
        if len(eigs_filt) == 0:
            continue
        
        # We'll store them
        file_data_list.append(eigs_filt)
        
        # For bin-edge calc, consider both E and -E
        all_energies_for_bins.append(eigs_filt)
        all_energies_for_bins.append(-eigs_filt)
    
    if not file_data_list:
        return None
    
    # Merge into single array
    all_energies_for_bins = np.concatenate(all_energies_for_bins)
    
    # 3) Compute unified bin edges
    bin_edges = compute_quantile_bin_edges(all_energies_for_bins, bin_frac=bin_frac)
    if len(bin_edges) < 2:
        return None
    
    n_bins = len(bin_edges) - 1
    
    # We'll gather r-values across all files, for each bin
    # So master_bin_rvals[b] will hold a list of all r-values from all files/trials
    master_bin_rvals = [[] for _ in range(n_bins)]
    
    # 4) For each file's data, do:
    #    - compute r-values for the "positive" eigenvalues,
    #    - compute r-values for the "negative" eigenvalues,
    #    - accumulate them in master_bin_rvals
    for eigs_filt in file_data_list:
        # Original trial
        file_result_pos = compute_rvals_in_bins(eigs_filt, bin_edges)
        # Negative trial
        eigs_neg = -eigs_filt
        file_result_neg = compute_rvals_in_bins(eigs_neg, bin_edges)
        
        # Accumulate into global
        for b in range(n_bins):
            if b in file_result_pos and len(file_result_pos[b]) > 0:
                master_bin_rvals[b].extend(file_result_pos[b])
            if b in file_result_neg and len(file_result_neg[b]) > 0:
                master_bin_rvals[b].extend(file_result_neg[b])
    
    # 5) Compute average and standard error in each bin
    avg_rvals = np.zeros(n_bins, dtype=float)
    std_errs  = np.zeros(n_bins, dtype=float)
    for b in range(n_bins):
        rvals_b = master_bin_rvals[b]
        if len(rvals_b) == 0:
            avg_rvals[b] = np.nan
            std_errs[b]  = np.nan
        else:
            mean_b = np.mean(rvals_b)
            std_b  = np.std(rvals_b, ddof=1)  # sample std
            avg_rvals[b] = mean_b
            std_errs[b]  = std_b / np.sqrt(len(rvals_b))  # SE
    
    # For plotting on an Energy axis, let's define bin_centers as the midpoint
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    results = {
        'bin_edges':   bin_edges,
        'bin_centers': bin_centers,
        'avg_rvals':   avg_rvals,
        'std_errs':    std_errs
    }
    return results

###############################################################################
# Main script: produce per-subdir figures + an overlay figure
###############################################################################

def main_unified_plots(folder_path, overlay_n_values, bin_frac=0.05):
    """
    1) Loop over subdirs that match "N=..." in overlay_n_values.
    2) For each subdir, produce one figure with 4 subplots (All, +1, -1, 0).
       Save it (or show it).
    3) Collect the results so we can overlay across N in a separate figure.
    4) Produce a single figure with 4 subplots, each subplot showing the
       data from all N overlaid (with error bars).
    """
    # Prepare storage so we can do the final overlay
    # We want to store something like: overlay_data[cfilter_label][N] = results
    cfilters = [
        ("All",  None),
        ("C=+1", 1),
        ("C=-1", -1),
        ("C=0",  0),
    ]
    overlay_data = {
        "All":  {},
        "C=+1": {},
        "C=-1": {},
        "C=0":  {}
    }
    
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    for subdir in subdirs:
        # e.g. "N=12_something"
        try:
            n_value = subdir.split('=')[-1].split('_')[0]
            n_value = int(n_value)
        except:
            continue
        
        if n_value not in overlay_n_values:
            continue
        
        print(f"=== Analyzing subdir: {subdir} (N={n_value}) ===")
        subdir_path = os.path.join(folder_path, subdir)
        
        # We'll build a figure with 4 subplots for the 4 c-filters
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()
        fig.suptitle(f"N={n_value}, bin_frac={bin_frac}", fontsize=14)
        
        for ax_idx, (label, cval) in enumerate(cfilters):
            ax = axes[ax_idx]
            # Analyze
            results = analyze_subdir_unified_bins(subdir_path,
                                                  bin_frac=bin_frac,
                                                  chern_filter=cval)
            if results is None:
                # Possibly no data or no .npz
                ax.set_title(f"{label}: no data")
                ax.set_xlabel("Energy")
                ax.set_ylabel("r")
                continue
            
            x = results['bin_centers']
            y = results['avg_rvals']
            yerr = results['std_errs']
            
            ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=3)
            ax.set_title(label)
            ax.set_xlabel("Energy (quantile bins)")
            ax.set_ylabel("Average r")
            
            # Also store it for the overlay
            overlay_data[label][n_value] = results
        
        plt.tight_layout()
        plt.show()  # or plt.savefig(f"N_{n_value}_4plots.png")
    
    # --------------------------------------------------------------
    # Now produce an OVERLAY figure across all N for each c-filter.
    # We'll create one figure with 4 subplots (one for each c-filter),
    # overlaying lines for each N in overlay_n_values.
    # --------------------------------------------------------------
    fig_overlay, axes_overlay = plt.subplots(2, 2, figsize=(10, 8))
    axes_overlay = axes_overlay.ravel()
    fig_overlay.suptitle("Overlay across all N", fontsize=14)
    
    for ax_idx, (label, _) in enumerate(cfilters):
        ax = axes_overlay[ax_idx]
        ax.set_title(label)
        ax.set_xlabel("Energy (quantile bins)")
        ax.set_ylabel("Average r")
        
        # Plot each N that we have
        for n_val in sorted(overlay_n_values):
            if n_val not in overlay_data[label]:
                continue
            res = overlay_data[label][n_val]
            x = res['bin_centers']
            y = res['avg_rvals']
            yerr = res['std_errs']
            ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=3,
                        label=f"N={n_val}")
        
        ax.legend()
    
    plt.tight_layout()
    plt.show()  # or plt.savefig("overlay_allN.png")

###############################################################################
# Example usage if running as script
###############################################################################
if __name__ == "__main__":
    folder_path = "/Users/eddiedeleu/Desktop/FinalData"  # adjust this path as needed
    overlay_n_values = [128,256,512,1024,2048]  # just an example
    main_unified_plots(folder_path, overlay_n_values, bin_frac=0.1)
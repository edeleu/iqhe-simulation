import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

###############################################################################
# Utility functions
###############################################################################

def calculate_r_values_within_window(window_eigs):
    """
    Calculate r-values for a single bin's subset of eigenvalues.
    r = min(s_{n}, s_{n+1}) / max(s_{n}, s_{n+1}),
    where s_{n} = E_{n} - E_{n-1}.
    We only compute r for interior points in the sorted array.
    """
    if len(window_eigs) < 3:
        return np.array([])  # Need at least 3 eigenvalues to compute meaningful r-values
    
    sorted_window = np.sort(window_eigs)
    diffs = np.diff(sorted_window)
    r_values = []
    # Calculate r-values for interior eigenvalues only
    for i in range(1, len(sorted_window) - 1):
        s_prev = diffs[i - 1]
        s_next = diffs[i]
        r = min(s_prev, s_next) / max(s_prev, s_next)
        r_values.append(r)
    
    return np.array(r_values)

def compute_quantile_bin_edges(all_energies, bin_frac=0.05):
    """
    Given a large array of energies (from ALL files + their negatives),
    compute a single set of bin edges so that each bin contains bin_frac
    fraction of the data by number count.
    e.g. bin_frac=0.05 => 20 bins total.

    Returns an array of bin_edges of length (n_bins+1).
    """
    if len(all_energies) < 3:
        return np.array([])
    
    sortedE = np.sort(all_energies)
    n_points = len(sortedE)
    n_bins = int(round(1.0 / bin_frac))  # e.g. 0.05 => 20 bins
    if n_bins < 1:
        n_bins = 1
    
    # quantiles in [0,1], e.g. 21 points for n_bins=20
    q = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(sortedE, q)
    return bin_edges

def compute_rvals_for_file(eigs, bin_edges):
    """
    Given the unified bin_edges and the eigenvalues for ONE “trial” (file),
    place them into bins, compute the r-values in each bin, and return a dict:
        { bin_index: list_of_r_vals }
    """
    if len(eigs) < 3 or len(bin_edges) < 2:
        return {}

    # Digitize to find which bin each E belongs to
    bin_indices = np.searchsorted(bin_edges, eigs, side='right') - 1
    # bin_indices[i] = b where bin_edges[b] <= E < bin_edges[b+1].
    
    n_bins = len(bin_edges) - 1
    result = {b: [] for b in range(n_bins)}
    
    # Group the E's by bin
    for b in range(n_bins):
        mask_b = (bin_indices == b)
        if not np.any(mask_b):
            continue
        # E's in bin b
        E_bin = eigs[mask_b]
        r_vals = calculate_r_values_within_window(E_bin)
        if len(r_vals) > 0:
            result[b].extend(r_vals)
    
    return result

def filter_chern(eigs, cvals, chern_filter):
    """
    Filter the eigenvalues E by cvals == chern_filter.
    If chern_filter=None => no filter => return all.
    If cvals is None => no filter => return all.
    If chern_filter is a single integer or list => use np.isin.
    """
    if chern_filter is None or cvals is None:
        return eigs
    if isinstance(chern_filter, int):
        chern_filter = [chern_filter]
    mask = np.isin(cvals, chern_filter)
    return eigs[mask]

###############################################################################
# Main Workflow
###############################################################################

def analyze_subdir_unified_bins(subdir_path, bin_frac=0.05, chern_filter=None):
    """
    1. Gather all (E) across all .npz in subdir (only applying chern_filter?),
       and also gather their negatives, to define a single set of bin edges.
    2. For each file, treat (E) as one "trial" and (-E) as another "trial";
       compute r-values in each bin using the unified bin_edges.
    3. Return a dictionary containing aggregated results across the subdir.
    """
    # 1) Load all .npz files from subdir
    file_list = []
    for f in os.listdir(subdir_path):
        if f.endswith('.npz'):
            file_list.append(os.path.join(subdir_path, f))
    if not file_list:
        print(f"No .npz files found in {subdir_path}.")
        return None
    
    # 2) Collect all E for bin-edge computation
    all_energies_for_bins = []
    
    # We'll store each file's data in a structure so we don't re-load it
    loaded_data = []
    
    for fpath in file_list:
        data = np.load(fpath)
        cvals = data['ChernNumbers'] if 'ChernNumbers' in data else None
        eigs = data['eigsPipi']
        
        # Filter by chern if desired
        eigs_filt = filter_chern(eigs, cvals, chern_filter)
        
        # We'll keep these for the next pass:
        loaded_data.append(eigs_filt)
        
        # For the bin edges, we consider both E and -E
        all_energies_for_bins.append(eigs_filt)
        all_energies_for_bins.append(-eigs_filt)
    
    if len(all_energies_for_bins) == 0:
        print(f"No data after filtering c={chern_filter} in {subdir_path}.")
        return None
    
    all_energies_for_bins = np.concatenate(all_energies_for_bins)
    
    # 3) Compute one set of unified bin edges
    bin_edges = compute_quantile_bin_edges(all_energies_for_bins, bin_frac=bin_frac)
    if len(bin_edges) < 2:
        print("Not enough data to form bin edges.")
        return None
    
    # 4) Now, for each file's eigenvalues, compute r-values in bins for:
    #    (a) the original (eigs_filt),
    #    (b) the negative ( -eigs_filt ).
    #    Then accumulate all these results across all files in subdir.
    
    n_bins = len(bin_edges) - 1
    # We'll keep a master list of r-values per bin across ALL trials:
    # trial = file plus "sign"
    master_bin_rvals = [ [] for _ in range(n_bins) ]
    
    for eigs_filt in loaded_data:
        # Original trial
        file_result_pos = compute_rvals_for_file(eigs_filt, bin_edges)
        # Negative trial
        eigs_neg = -eigs_filt
        file_result_neg = compute_rvals_for_file(eigs_neg, bin_edges)
        
        # Accumulate both sets
        # file_result_*.keys() = bin indices, each mapping to a list of r vals
        for b in range(n_bins):
            # For the positive
            if b in file_result_pos and len(file_result_pos[b]) > 0:
                master_bin_rvals[b].extend(file_result_pos[b])
            # For the negative
            if b in file_result_neg and len(file_result_neg[b]) > 0:
                master_bin_rvals[b].extend(file_result_neg[b])
    
    # 5) We can compute average r-value in each bin
    avg_rvals = []
    for b in range(n_bins):
        if len(master_bin_rvals[b]) == 0:
            avg_rvals.append(np.nan)
        else:
            avg_rvals.append(np.mean(master_bin_rvals[b]))
    
    # Optionally compute a representative "center" of each bin if you like
    # (we can use the midpoint of bin_edges or the median of energies in that bin).
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Return the info we need to plot
    results_dict = {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'avg_rvals': np.array(avg_rvals),
        'bin_frac': bin_frac,
        'chern_filter': chern_filter
    }
    return results_dict

def main_unified_plots(folder_path, overlay_n_values, bin_frac=0.05):
    """
    Example main function that:
      - Iterates over subdirs named "N=..." inside folder_path,
        ensuring that int(n_value) is in overlay_n_values.
      - For each subdir, we produce 4 plots: All Chern, C=+1, C=-1, C=0.
      - You can adapt if you only want a single figure or something else.
    """
    subdirs = [d for d in os.listdir(folder_path)
               if os.path.isdir(os.path.join(folder_path, d))]
    
    # We'll do a big loop, but you can adapt to your exact plotting style.
    for subdir in subdirs:
        # Example subdir name: "N=12_something"
        try:
            n_value = subdir.split('=')[-1].split('_')[0]
            n_value = int(n_value)
        except:
            continue
        
        if n_value not in overlay_n_values:
            continue
        
        print(f"\n=== Analyzing subdir: {subdir} (N={n_value}) ===")
        subdir_path = os.path.join(folder_path, subdir)
        
        # We'll build a figure with 2x2 subplots for the 4 Chern filters
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()
        cfilters = [
            ("All Chern",  None),
            ("C=+1",       1),
            ("C=-1",      -1),
            ("C=0",        0),
        ]
        
        for ax_idx, (label, cfilt) in enumerate(cfilters):
            print(f"  -> Chern filter = {label}")
            result = analyze_subdir_unified_bins(subdir_path,
                                                 bin_frac=bin_frac,
                                                 chern_filter=cfilt)
            if result is None:
                print("No data or not enough data to form bins. Skipping plot.")
                continue
            
            bin_centers = result['bin_centers']
            avg_rvals   = result['avg_rvals']
            
            ax = axes[ax_idx]
            ax.plot(bin_centers, avg_rvals, 'o-')
            ax.set_title(label)
            ax.set_xlabel("Energy (unified quantile bins)")
            ax.set_ylabel("Average r")
        
        plt.suptitle(f"Subdir: {subdir}, bin_frac={bin_frac}")
        plt.tight_layout()
        plt.show()

###############################################################################
# Example usage (if you run this script directly)
###############################################################################
if __name__ == "__main__":
    # Example: folder_path = "/path/to/data"
    folder_path = "/Users/eddiedeleu/Desktop/FinalData"  # adjust this path as needed
    overlay_n_values = [64,128,256,512,1024,2048]  # just an example
    main_unified_plots(folder_path, overlay_n_values, bin_frac=0.05)

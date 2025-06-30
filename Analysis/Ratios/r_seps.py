import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.stats import sem
from matplotlib import gridspec, rc
from scipy.interpolate import make_interp_spline

# Configure plot settings
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": (8, 5),          # Default figure size for single-column plots
    "figure.dpi": 300,
    "lines.linewidth": 1,
    "grid.color": "gray",
    "grid.alpha": 0.6,
    "axes.grid": True,
    "legend.loc": "best",
})
rc('text.latex', preamble=r'\usepackage{amsmath}')

def load_files(folder_path):
    """Load all .npz files in directory"""
    valid_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    return [os.path.join(folder_path, f) for f in valid_files]

def calculate_r_values_within_window(window_eigs):
    """Calculate r-values for a single energy window"""
    if len(window_eigs) < 3:
        return np.array([])  # Need at least 3 eigenvalues to compute r-values
    
    diffs = np.diff(window_eigs)
    r_values = []
    
    # Calculate r-values for interior eigenvalues only
    for i in range(1, len(window_eigs)-1):
        s_prev = diffs[i-1]
        s_next = diffs[i]
        r = min(s_prev, s_next)/max(s_prev, s_next)
        r_values.append(r)
    
    return np.array(r_values)

def create_centered_windows(max_energy, window_width=0.5):
    """Create energy windows with 0 as an explicit center"""
    half_width = window_width / 2
    max_energy_rounded = np.ceil((max_energy + half_width) / window_width) * window_width
    
    # Create positive centers first (including 0)
    positive_centers = np.arange(0, max_energy_rounded + half_width, window_width)
    
    # Create full centers array (negative + 0 + positive)
    centers = np.concatenate([-positive_centers[1:][::-1], positive_centers])
    
    # Create edges from centers
    edges = np.concatenate([centers - half_width, [centers[-1] + half_width]])
    
    return centers, edges

def analyze_file(eigs, edges):
    """Analyze a single file's eigenvalues"""
    sorted_eigs = np.sort(eigs)
    indices = np.digitize(sorted_eigs, edges)
    
    file_results = {}
    for idx in range(len(edges)-1):
        # Get eigenvalues in this window
        mask = (indices == idx+1)  # digitize returns 1-based indices
        window_eigs = sorted_eigs[mask]
        
        # Calculate r-values within window
        r_values = calculate_r_values_within_window(window_eigs)
        
        # Store results
        if len(r_values) > 0:
            file_results[idx] = r_values
    
    return file_results

# def symmetrize_r_values(results):
#     """Copy r-values from positive windows to negative windows and set them equal."""
#     num_windows = len(results['centers'])
#     for idx in range(num_windows // 2):  # Only iterate over half of the windows
#         mirror_idx = num_windows - idx - 1  # Find the mirror index
        
#         # Combine r-values from positive and negative windows
#         combined_r_values = np.concatenate((results['window_data'][idx], results['window_data'][mirror_idx]))
#         # print("Length of results['window_data'][idx]:", len(results['window_data'][idx]))
#         # print("Length of results['window_data'][mirror_idx]:", len(results['window_data'][mirror_idx]))

#         # Set both windows to the same combined list
#         results['window_data'][idx] = combined_r_values
#         results['window_data'][mirror_idx] = combined_r_values
#         # print("Length of combined_r_values:", len(combined_r_values))
#         # Handle the central window at E = 0 (if num_windows is odd, it has a unique center)
#     if num_windows % 2 == 1:
#         central_idx = num_windows // 2
#         results['window_data'][central_idx] = np.concatenate((results['window_data'][central_idx], results['window_data'][central_idx]))

#     return results


def analyze_all_files(file_list, centers, edges):
    """Analyze all files and collect r-values in windows"""
    results = {
        'centers': centers,
        'window_data': {i: [] for i in range(len(centers))}
    }
    
    for fpath in tqdm(file_list, desc="Processing files"):
        data = np.load(fpath)
        eigs = data['eigsPipi']  # Adjust key if needed
        c_mask = np.isclose(data["ChernNumbers"], 1, atol=1e-5)

        eigs = eigs[c_mask]

        # Get per-window r-values for this file
        file_results = analyze_file(eigs, edges)
        
        # Aggregate results
        for idx, r_values in file_results.items():
            results['window_data'][idx].extend(r_values)

        # do the entire thing again, for symmetry
        negEigs = -eigs

        # Get per-window r-values for this file
        file_results = analyze_file(negEigs, edges)
        
        # Aggregate results
        for idx, r_values in file_results.items():
            results['window_data'][idx].extend(r_values)

    # results = symmetrize_r_values(results)

    return results

def calculate_statistics(results):
    """Calculate statistics for each window"""
    stats = {
        'centers': [],
        'mean_r': [],
        'std_r': [],
        'sem_r': [],
        'counts': []
    }
    
    for idx in range(len(results['centers'])):
        r_values = results['window_data'][idx]
        if len(r_values) == 0:
            continue
            
        stats['centers'].append(results['centers'][idx])
        stats['mean_r'].append(np.mean(r_values))
        stats['std_r'].append(np.std(r_values, ddof=1))
        stats['sem_r'].append(sem(r_values))
        stats['counts'].append(len(r_values))
    
    return stats

def plot_results(stats, window_width=0.5, min_error=2e-4, n_value=""):
    fig, ax = plt.subplots()

    # Filter out windows with insufficient data
    valid = np.array(stats['counts']) > 100  # Require at least 100 r-values
    centers = np.array(stats['centers'])[valid]
    mean_r = np.array(stats['mean_r'])[valid]
    sem_r = np.array(stats['sem_r'])[valid]
    
    # Plot all data points
    ax.plot(centers, mean_r, 'o', markersize=6, label=f'Data (window width: {window_width})')
    
    # Plot error bars only for points with error above threshold
    mask = sem_r > min_error
    ax.errorbar(centers[mask], mean_r[mask], yerr=sem_r[mask], fmt='none', ecolor='gray', capsize=5)
    
    # Create smooth line of best fit
    X_smooth = np.linspace(centers.min(), centers.max(), 300)
    spl = make_interp_spline(centers, mean_r, k=3)
    y_smooth = spl(X_smooth)
    
    # Plot smooth line
    ax.plot(X_smooth, y_smooth, color='red', label='Smooth Spline Fit')
    
    ax.set_xlabel('Energy (E)', fontsize=14)
    ax.set_ylabel(r'Average $\langle r \rangle$', fontsize=14)
    ax.set_title(f'Energy-Dependent r Statistics ({n_value})', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'windowed_r_analysis_{n_value}.pdf')
    plt.close()


def create_overlay_plot(all_stats, n_values, window_width=0.5, min_error=5e-4):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(n_values)))

    for i, (n_value, stats) in enumerate(zip(n_values, all_stats)):
        try:
            valid = np.array(stats['counts']) > 50
            centers = np.array(stats['centers'])[valid]
            mean_r = np.array(stats['mean_r'])[valid]
            sem_r = np.array(stats['sem_r'])[valid]
            
            # Plot data points and smooth fit
            ax.plot(centers, mean_r, 'o', markersize=4, color=colors[i], alpha=0.5)
            
            mask = sem_r > min_error
            ax.errorbar(centers[mask], mean_r[mask], yerr=sem_r[mask], fmt='none', ecolor='gray', capsize=2)

            # Create smooth line of best fit
            X_smooth = np.linspace(centers.min(), centers.max(), 300)
            spl = make_interp_spline(centers, mean_r, k=3)
            y_smooth = spl(X_smooth)
            
            ax.plot(X_smooth, y_smooth, color=colors[i], label=f'N = {n_value}')
        except:
            print("None at", n_value)

    ax.axhline(0.3863, color='red', linestyle='--', label='Poisson Value 0.3863')
    ax.axhline(0.5996, color='green', linestyle='--', label='GUE Value 0.5996')
    # ax.axhline(0.6744, color='blue', linestyle='--', label='GSE Value 0.6744')

    ax.set_xlabel('Energy (E)', fontsize=14)
    ax.set_ylabel(r'Average $\langle r \rangle$', fontsize=14)
    ax.set_title(f'Energy-Dependent r Statistics Comparison', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'windowed_r_analysis_comparison_Chern1_New2.pdf')
    plt.close()

def main_analysis(folder_path, window_dict, overlay_n_values=[64,128,256,512,1024,2048]):
    all_stats = []
    processed_n_values = []

    # Get list of all subdirectories
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    for subdir in subdirs:
        n_value = subdir.split('=')[-1].split('_')[0]  # Extract numeric value from directory name
        if int(n_value) not in overlay_n_values:
            continue

        subdir_path = os.path.join(folder_path, subdir)
        
        # Initialize analysis for this subdirectory
        file_list = load_files(subdir_path)
        
        if not file_list:
            print(f"No .npz files found in {subdir_path}. Skipping.")
            continue
        
        # First pass to determine maximum energy
        print(f"Determining energy range for N={n_value}...")
        all_eigs = []
        for fpath in tqdm(file_list):
            data = np.load(fpath)
            all_eigs.extend(data['eigsPipi'])
        max_energy = np.max(np.abs(all_eigs))
        
        # Create centered windows
        window_width = window_dict.get(int(n_value))
        centers, edges = create_centered_windows(max_energy, window_width)
        
        # Analyze files
        results = analyze_all_files(file_list, centers, edges)
        stats = calculate_statistics(results)
        
        # Store stats for overlay plot
        all_stats.append(stats)
        processed_n_values.append(n_value)
        
        # Create individual plot
        # plot_results(stats, window_width, n_value=f'N={n_value}')
        
        print(f"Completed analysis for N={n_value}")

    # Create overlay plot
    create_overlay_plot(all_stats, processed_n_values, window_width)

def plot_r_vs_N_at_energy(folder_path, energy_value=0.0, window_widths=[0.5, 0.1, 0.05], min_counts=100):
    """
    For each specified window width, analyze all subdirectories (each for a given N)
    in folder_path, and extract the average r-value (and SEM) at the window closest to energy_value.
    Then plot r vs. N with different curves for each window width.
    
    Parameters:
        folder_path (str): Path to the parent directory containing subdirectories.
        energy_value (float): The energy at which to extract r-value statistics.
        window_widths (list): List of window sizes to use for the analysis.
        min_counts (int): Minimum number of r-values required in a window to include the point.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    # Prepare a dictionary to store results for each window width.
    # Each key will map to a list of tuples: (N, mean_r, sem_r)
    results_by_window = {w: [] for w in window_widths}
    
    # Get list of subdirectories (each corresponding to a system size N)
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    # Loop over subdirectories (each representing a different N value)
    for subdir in tqdm(subdirs, desc="Processing subdirectories"):
        subdir_path = os.path.join(folder_path, subdir)
        # Assume that the subdirectory name contains an "N=" identifier.
        try:
            n_value = int(subdir.split('=')[-1].split('_')[0])
        except Exception as e:
            print(f"Could not parse N value from {subdir}: {e}")
            continue
        
        # Load files from this subdirectory
        file_list = load_files(subdir_path)
        if not file_list:
            print(f"No .npz files found in {subdir_path}. Skipping.")
            continue
        
        # For each window width, perform the analysis for this N value.
        for window_width in window_widths:
            # Determine maximum energy over all files
            max_energy = 6
            
            # Create centered windows for this window width
            centers, edges = create_centered_windows(max_energy, window_width)
            
            # Analyze the files using the existing pipeline
            results = analyze_all_files(file_list, centers, edges)
            stats = calculate_statistics(results)
            
            # Find the window (index) whose center is closest to energy_value
            centers_arr = np.array(stats['centers'])
            diff = np.abs(centers_arr - energy_value)
            if len(diff) == 0:
                continue
            target_idx = diff.argmin()
            
            # Get mean r and sem for that window, if there is sufficient data
            if stats['counts'][target_idx] >= min_counts:
                mean_r = stats['mean_r'][target_idx]
                sem_r = stats['sem_r'][target_idx]
                results_by_window[window_width].append((n_value, mean_r, sem_r))
    
    # Now, create the plot for r vs. N for each window width.
    plt.figure(figsize=(8, 6))
    markers = ['o', 's', '^', 'd', '*']  # Some marker options
    for i, window_width in enumerate(window_widths):
        # Extract and sort by N
        data = results_by_window[window_width]
        if not data:
            continue
        data.sort(key=lambda x: x[0])  # sort by N
        n_vals, mean_rs, sem_rs = zip(*data)
        n_vals = np.array(n_vals)
        mean_rs = np.array(mean_rs)
        sem_rs = np.array(sem_rs)
        
        plt.errorbar(n_vals, mean_rs, yerr=sem_rs, fmt=markers[i % len(markers)], capsize=5,
                     label=f'Window width = {window_width}')
    
    plt.xlabel('System Size, N', fontsize=12)
    plt.ylabel(r'Average $\langle r \rangle$', fontsize=12)
    plt.title(r'$\langle r \rangle$ vs. N at $E=%g$' % energy_value, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'r_vs_N_at_E{energy_value}_C_All.pdf')
    plt.show()


# Example usage:
if __name__ == "__main__":
    folder_path = "/Users/eddiedeleu/Desktop/FinalData"  # adjust this path as needed
    # Plot r vs. N at energy E=0 for window widths 0.5, 0.1, and 0.05
    # plot_r_vs_N_at_energy(folder_path, energy_value=1.5, window_widths=[0.5, 0.1, 0.05])

    # Example usage
    # window_dict = {2048: 0.025, 1024:0.05,512:0.1,256:0.2,128:0.4,64:0.8}
    # window_dict = {2048: 0.02, 1024:0.04,512:0.08,256:0.16,128:0.32,64:0.64}
    window_dict = {2048: 0.04, 1024:0.08,512:0.16,256:0.32,128:0.64,64:1.6}

    main_analysis("/Users/eddiedeleu/Desktop/FinalData", window_dict)

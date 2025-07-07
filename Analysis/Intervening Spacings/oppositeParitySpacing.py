import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import rc

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
    "axes.grid": False
})

rc('text.latex', preamble=r'\usepackage{amsmath}')

def intervening_opposite_parity(eigs, cherns, source_val, target_val):
    sort_idx = np.argsort(eigs)
    eigs = eigs[sort_idx]
    cherns = cherns[sort_idx]

    intervening_counts = []
    energy_separations = []

    for i, c in enumerate(cherns):
        if c != source_val:
            continue

        best_distance = np.inf
        best_gap = None

        # Search left
        gap = 0
        for j in range(i - 1, -1, -1):
            gap += 1
            if cherns[j] == target_val:
                dist = abs(eigs[j] - eigs[i])
                best_distance = dist
                best_gap = gap - 1
                break

        # Search right
        gap = 0
        for j in range(i + 1, len(cherns)):
            gap += 1
            if cherns[j] == target_val:
                dist = abs(eigs[j] - eigs[i])
                if dist < best_distance:
                    best_distance = dist
                    best_gap = gap - 1
                break

        if best_gap is not None:
            intervening_counts.append(best_gap)
            energy_separations.append(best_distance)

    return intervening_counts, energy_separations

def analyze_opposite_parity(folder_path, output_csv='opposite_parity_stats.csv', hist_dir='opposite_parity_histograms'):
    os.makedirs(hist_dir, exist_ok=True)
    files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    print(f"Processing {len(files)} files.")

    results = {
        '+1_to_-1': {'gaps': [], 'seps': []},
        '-1_to_+1': {'gaps': [], 'seps': []},
        '-1_to_0': {'gaps': [], 'seps': []},
        '+1_to_0': {'gaps': [], 'seps': []}
    }

    for idx, f in enumerate(files, 1):
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(files)} files.")

        data = np.load(os.path.join(folder_path, f))
        if not np.isclose(data['SumChernNumbers'], 1, atol=1e-5):
            continue

        cherns = data['ChernNumbers']
        eigs = data['eigsPipi']
        sorted_indices = np.argsort(eigs)
        sorted_cherns = cherns[sorted_indices]
        sorted_eigs = eigs[sorted_indices]

        ## MASK SECTION
        mask = (sorted_eigs >= -0.03) & (sorted_eigs <= 0.03)
        cherns = sorted_cherns[mask]
        eigs = sorted_eigs[mask]
        

        for source, target, key in [(1, -1, '+1_to_-1'), (-1, 1, '-1_to_+1'), (-1,0,'-1_to_0'), (+1,0,'+1_to_0')]:
            gaps, seps = intervening_opposite_parity(eigs, cherns, source, target)
            results[key]['gaps'].extend(gaps)
            results[key]['seps'].extend(seps)

    # Compile results
    rows = []
    for key, data in results.items():
        gaps = data['gaps']
        seps = data['seps']
        count = len(gaps)
        avg_gap = np.mean(gaps) if gaps else np.nan
        avg_sep = np.mean(seps) if seps else np.nan

        rows.append({
            'Pair Type': key,
            'Count': count,
            'Average Number of Intervening States': avg_gap,
            'Average Energy Separation': avg_sep
        })

        if gaps:
            plt.figure()
            plt.hist(gaps, bins=range(0, max(gaps)+2), align='left', edgecolor='black')
            plt.xlabel('Number of Intervening States')
            plt.ylabel('Frequency')
            plt.title(f'Intervening States: Chern {key}')
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, f'gap_hist_{key}.pdf'))
            plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv} and histograms to {hist_dir}/")

# Example usage
analyze_opposite_parity("/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/N=1024_Mem/")
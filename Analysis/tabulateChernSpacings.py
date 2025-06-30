import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
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
def count_intermediate_cherns(chern_array, eig_array, target):
    indices = np.where(chern_array == target)[0]
    gaps = []
    separations = []
    for i in range(len(indices) - 1):
        idx1 = indices[i]
        idx2 = indices[i+1]
        gap = idx2 - idx1 - 1  # number of entries in between
        if gap >= 0:
            gaps.append(gap)
            separations.append(abs(eig_array[idx2] - eig_array[idx1]))
    return gaps, separations

def analyze_all_targets(folder_path, output_csv='chern_gap_stats.csv', hist_dir='chern_gap_histograms'):
    os.makedirs(hist_dir, exist_ok=True)
    files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    print(f"Processing {len(files)} files.")

    chern_targets = range(-3, 4)
    gap_records = defaultdict(list)
    sep_records = defaultdict(list)

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

        for target in chern_targets:
            gaps, seps = count_intermediate_cherns(sorted_cherns, sorted_eigs, target)
            gap_records[target].extend(gaps)
            sep_records[target].extend(seps)

    # Summarize results
    rows = []
    for target in chern_targets:
        gaps = gap_records[target]
        seps = sep_records[target]
        count = len(gaps)
        avg_gap = np.mean(gaps) if gaps else np.nan
        avg_sep = np.mean(seps) if seps else np.nan

        row = {
            'Chern Number': target,
            'Count': count,
            'Average Number of Intervening States': avg_gap,
            'Average Energy Separation': avg_sep
        }

        rows.append(row)

        # Save histogram
        if gaps:
            plt.figure()
            plt.hist(gaps, bins=range(0, max(gaps)+2), align='left', edgecolor='black')
            plt.xlabel('Number of Intervening States')
            plt.ylabel('Frequency')
            plt.title(f'Histogram for Number of Intervening States for Chern {target}')
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, f'gap_hist_chern_{target}.pdf'))
            plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved gap statistics to {output_csv}")
    print(f"Saved individual histograms to {hist_dir}/")

# Example usage
analyze_all_targets("/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/N=1024_Mem/")
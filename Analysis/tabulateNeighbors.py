import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.colors as mcolors

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

def tabulate_adjacent_cherns(folder_path, output_csv='chern_adjacency_counts.csv', output_plot='chern_adjacency_heatmap.pdf'):
    files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    total = len(files)
    print(f"Processing {total} files.")

    adjacency_counts = {}

    for idx, f in enumerate(files, 1):
        if idx % 100 == 0:
            print(f"Processed {idx}/{total} files.")

        data = np.load(os.path.join(folder_path, f))
        if not np.isclose(data['SumChernNumbers'], 1, atol=1e-5):
            continue

        cherns = data['ChernNumbers']

        sorted_indices = np.argsort(data['eigsPipi'])
        sorted_cherns = cherns[sorted_indices]

        for i in range(len(sorted_cherns) - 1):
            chern_a, chern_b = sorted_cherns[i], sorted_cherns[i + 1]

            adjacency_counts[(chern_a, chern_b)] = adjacency_counts.get((chern_a, chern_b), 0) + 1
            adjacency_counts[(chern_b, chern_a)] = adjacency_counts.get((chern_b, chern_a), 0) + 1

    df = pd.DataFrame.from_dict(adjacency_counts, orient='index', columns=['count']).reset_index()
    df[['Chern A', 'Chern B']] = pd.DataFrame(df['index'].tolist(), index=df.index)
    df_pivot = df.pivot_table(index='Chern A', columns='Chern B', values='count', fill_value=0)

    df_pivot.to_csv(output_csv)
    print(f"\nAdjacency counts saved to {output_csv}")

    fig, ax = plt.subplots(figsize=(8, 6))
    norm = mcolors.LogNorm(vmin=df_pivot.values[df_pivot.values > 0].min(), vmax=df_pivot.values.max())
    im = ax.imshow(df_pivot.values, cmap='Blues', norm=norm, origin='lower')

    # Get tick labels
    tick_labels_x = df_pivot.columns.values
    tick_labels_y = df_pivot.index.values

    # Set tick labels centered on cells
    ax.set_xticks(np.arange(len(tick_labels_x)))
    ax.set_yticks(np.arange(len(tick_labels_y)))
    ax.set_xticklabels(tick_labels_x)
    ax.set_yticklabels(tick_labels_y)

    # Draw grid lines on cell edges (integer boundaries)
    ax.set_xticks(np.arange(-0.5, len(tick_labels_x), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tick_labels_y), 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.set_xlabel('Adjacent Chern Number')
    ax.set_ylabel('Reference Chern Number')
    ax.set_title('Chern Number Adjacency Heatmap')

    cbar = plt.colorbar(im, ax=ax, label='Log-scaled Count')
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)

    print(f"Heatmap plot saved to {output_plot}")

# Run your adjacency analysis
tabulate_adjacent_cherns("/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/N=1024_Mem/")

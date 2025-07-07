import numpy as np
import os

# Include the nearest_nonzero_chern_stats function defined previously here
def nearest_nonzero_chern_stats(eigs, cherns):
    sort_indices = np.argsort(eigs)
    sorted_eigs = eigs[sort_indices]
    sorted_cherns = cherns[sort_indices]

    ## MASK SECTION
    mask = (sorted_eigs >= -0.03) & (sorted_eigs <= 0.03)
    sorted_cherns = sorted_cherns[mask]
    sorted_eigs = sorted_eigs[mask]

    stats = {
        '+1': {'nearest +1': 0, 'nearest -1': 0},
        '-1': {'nearest +1': 0, 'nearest -1': 0}
    }

    n = len(sorted_eigs)

    for i in range(n):
        current_chern = sorted_cherns[i]
        if current_chern not in [-1, 1]:
            continue

        nearest_distance = np.inf
        nearest_chern = None

        j = i - 1
        while j >= 0:
            if sorted_cherns[j] != 0:
                nearest_distance = sorted_eigs[i] - sorted_eigs[j]
                nearest_chern = sorted_cherns[j]
                break
            j -= 1

        j = i + 1
        while j < n:
            if sorted_cherns[j] != 0:
                right_distance = sorted_eigs[j] - sorted_eigs[i]
                if right_distance < nearest_distance:
                    nearest_distance = right_distance
                    nearest_chern = sorted_cherns[j]
                break
            j += 1

        key = '+1' if current_chern == 1 else '-1'
        if nearest_chern == 1:
            stats[key]['nearest +1'] += 1
        elif nearest_chern == -1:
            stats[key]['nearest -1'] += 1

    return stats


def analyze_nearest_cherns(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    total = len(files)
    print(f"Processing {total} files.")

    cumulative_stats = {
        '+1': {'nearest +1': 0, 'nearest -1': 0},
        '-1': {'nearest +1': 0, 'nearest -1': 0}
    }

    for idx, f in enumerate(files, 1):
        if idx % 100 == 0:
            print(f"Processed {idx}/{total} files.")

        data = np.load(os.path.join(folder_path, f))
        if not np.isclose(data['SumChernNumbers'], 1, atol=1e-5):
            continue

        eigs = data['eigsPipi']
        cherns = data['ChernNumbers']

        stats = nearest_nonzero_chern_stats(eigs, cherns)
        for chern_val in ['+1', '-1']:
            for neighbor in ['nearest +1', 'nearest -1']:
                cumulative_stats[chern_val][neighbor] += stats[chern_val][neighbor]

    print("\nFinal statistics:")
    for chern, neighbors in cumulative_stats.items():
        print(f"Chern {chern}:")
        for neighbor_type, count in neighbors.items():
            print(f"  {neighbor_type}: {count}")

# Run your analysis
analyze_nearest_cherns("/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/N=1024_Mem/")

import scipy.stats as stats
import os
import numpy as np
import csv
import timeit
from numba import njit, prange, jit
from timeit import default_timer as timer
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib import rc
from collections import Counter

def load_trial_data(file_path):
    """Load trial data efficiently from a .npz file."""
    data = np.load(file_path)
    return {
        "PotentialMatrix": data["PotentialMatrix"],
        "ChernNumbers": data["ChernNumbers"],
        "SumChernNumbers": data["SumChernNumbers"],
        "eigs00": data["eigs00"],
        "eigs0pi": data["eigs0pi"],
        "eigsPi0": data["eigsPi0"],
        "eigsPipi": data["eigsPipi"]
    }

folderPath = "/Users/eddiedeleu/Downloads/N=16"
chern_counts = Counter()
nonzero_counts=[]

for file_name in sorted(os.listdir(folderPath)):
    if file_name.endswith(".npz"):
        file_path = os.path.join(folderPath, file_name)
        datas = load_trial_data(file_path)

        # Only process files where the sum of Chern numbers is close to 1
        if np.isclose(datas["SumChernNumbers"], 1, atol=1e-5):
            currentCherns = datas["ChernNumbers"]  # Array of Chern numbers in this file

            # Update counts for all unique Chern numbers
            chern_counts.update(currentCherns)
            nonzero_count = np.count_nonzero(currentCherns)
            nonzero_counts.append(nonzero_count)


# Print final counts
for chern_number, count in sorted(chern_counts.items()):
    print(f"Chern={chern_number}: count={count}")

# Compute statistics
average_nonzero = np.mean(nonzero_counts)
std_nonzero = np.std(nonzero_counts, ddof=1)  # Sample standard deviation

# Print results
print(f"Average nonzero Chern states per file: {average_nonzero}")
print(f"Standard deviation: {std_nonzero}")



## NOW let's try plotting everything
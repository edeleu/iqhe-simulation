import os
import numpy as np
import hashlib
from collections import defaultdict
from tqdm import tqdm

# Base directory
base_dir = "/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData"

# Dictionary to store unique hashes per system size
hash_dict = defaultdict(set)
duplicate_records = []

def hash_matrix(matrix):
    """Compute SHA-256 hash of a NumPy array."""
    return hashlib.sha256(matrix.tobytes()).hexdigest()

# Traverse subdirectories by system size
for system_size_folder in sorted(os.listdir(base_dir)):
    system_path = os.path.join(base_dir, system_size_folder)
    if not os.path.isdir(system_path):
        continue

    # Gather .npz files in this folder
    npz_files = [f for f in os.listdir(system_path) if f.endswith(".npz")]
    if not npz_files:
        continue

    print(f"\nProcessing system size folder: {system_size_folder} ({len(npz_files)} files)")
    for file in tqdm(npz_files, desc=f"Checking {system_size_folder}", unit="file"):
        full_path = os.path.join(system_path, file)
        try:
            data = np.load(full_path)
            if "PotentialMatrix" in data:
                matrix = data["PotentialMatrix"]
                mat_hash = hash_matrix(matrix)
                matrix_shape = matrix.shape

                if mat_hash in hash_dict[matrix_shape]:
                    duplicate_records.append((matrix_shape, full_path))
                    tqdm.write(f"Duplicate found: {full_path}")
                else:
                    hash_dict[matrix_shape].add(mat_hash)
        except Exception as e:
            tqdm.write(f"Error loading {full_path}: {e}")

# Summary output
print("\n=== Duplicate Summary ===")
if not duplicate_records:
    print("No duplicates found.")
else:
    for shape, path in duplicate_records:
        print(f"Duplicate in system size {shape}: {path}")

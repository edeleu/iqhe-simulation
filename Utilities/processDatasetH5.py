import os
import numpy as np
import h5py
from tqdm import tqdm

def convert_system_size_to_hdf5(folder_path, output_file):
    """Convert all .npz files in a system-size folder into one .h5 file (pre-load into memory)."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    files.sort()
    n_trials = len(files)
    if n_trials == 0:
        print(f"No .npz files found in {folder_path}")
        return

    print(f"Loading {n_trials} trials into memory from {folder_path} ...")

    # Load everything into memory first
    data_all = {}
    filenames = []

    # Open one file to discover structure
    sample = np.load(os.path.join(folder_path, files[0]))
    keys = list(sample.keys())

    # Initialize in-memory lists
    for k in keys:
        data_all[k] = []

    # Read all .npz files
    for fname in tqdm(files, desc="Preloading trials"):
        arrs = np.load(os.path.join(folder_path, fname))
        filenames.append(fname)
        for k in keys:
            data_all[k].append(arrs[k])

    # Convert lists to stacked arrays
    for k in keys:
        data_all[k] = np.stack(data_all[k], axis=0)  # shape (n_trials, ...)

    print("✅ Finished loading into memory. Now writing HDF5 file...")

    # Write everything in one pass to HDF5
    with h5py.File(output_file, "w") as f_out:
        # Filenames dataset (UTF-8 strings)
        dt = h5py.string_dtype(encoding="utf-8")
        f_out.create_dataset("filenames", data=filenames, dtype=dt)

        # Create and write each array dataset
        for k, arr in data_all.items():
            f_out.create_dataset(
                k,
                data=arr,
                compression="gzip",  # fast, portable compression
                chunks=True
            )

    print(f"✅ Saved {n_trials} trials into {output_file}")


if __name__ == "__main__":
    base_path = "/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/"
    system_sizes = [1024, 2048,512,256]

    for n in system_sizes:
        if n in [1024, 2048]:
            folder_path = os.path.join(base_path, f"N={n}_Mem")
        else:
            folder_path = os.path.join(base_path, f"N={n}")

        if not os.path.exists(folder_path):
            print(f"⚠️ Directory {folder_path} not found, skipping.")
            continue

        output_file = os.path.join(base_path, f"N={n}_GZIP.h5")
        convert_system_size_to_hdf5(folder_path, output_file)

import os
import numpy as np
import csv
from tqdm import tqdm

def load_trial_data(file_path):
    data = np.load(file_path)
    return {
        "ChernNumbers": data["ChernNumbers"],
        "SumChernNumbers": data["SumChernNumbers"],
        "eigs00": data["eigs00"],
        "eigs0pi": data["eigs0pi"],
        "eigsPi0": data["eigsPi0"],
        "eigsPipi": data["eigsPipi"]
    }

def export_csv(folder_path, system_size, output_dir):
    files = [f for f in os.listdir(folder_path) if f.endswith(".npz")]
    files.sort()
    
    csv_all_path = os.path.join(output_dir, f"N={system_size}_All.csv")
    csv_subset_path = os.path.join(output_dir, f"N={system_size}_EBelow0.25.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open CSV writers
    with open(csv_all_path, "w", newline="") as f_all, open(csv_subset_path, "w", newline="") as f_sub:
        writer_all = csv.writer(f_all)
        writer_sub = csv.writer(f_sub)
        
        # Write headers
        header = ["trial#", "eigs00", "eigsPi0", "eigs0Pi", "eigsPiPi", "chern"]
        writer_all.writerow(header)
        writer_sub.writerow(header)
        
        # Iterate over trials
        for trial_idx, fname in enumerate(tqdm(files, desc=f"Processing N={system_size}")):
            data = load_trial_data(os.path.join(folder_path, fname))
            
            sum_chern = data["SumChernNumbers"]
            if not np.isclose(sum_chern, 1.0, atol=1e-5):
                continue  # skip trials not meeting SumChernNumbers=1
            
            # For each eigenvalue index
            N = len(data["eigs00"])
            for i in range(N):
                row = [
                    trial_idx + 1,  # trial number 1-based
                    data["eigs00"][i],
                    data["eigsPi0"][i],
                    data["eigs0pi"][i],
                    data["eigsPipi"][i],
                    data["ChernNumbers"][i]
                ]
                writer_all.writerow(row)
                
                # Check if any of the 4 eigenvalues < 0.25
                if any(np.abs(e) < 0.25 for e in row[1:5]):
                    writer_sub.writerow(row)

    print(f"✅ CSV files written:\n - {csv_all_path}\n - {csv_subset_path}")


if __name__ == "__main__":
    base_path = "/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData/"
    output_dir = "/scratch/gpfs/ed5754/iqheFiles/CSV_Output/"
    
    system_sizes = [64,96,128,192,256,512,1024,2048]
    
    for n in system_sizes:
        folder_path = os.path.join(base_path, f"N={n}_Mem" if n in [1024, 2048] else f"N={n}")
        if not os.path.exists(folder_path):
            print(f"Directory {folder_path} not found, skipping.")
            continue
        export_csv(folder_path, n, output_dir)

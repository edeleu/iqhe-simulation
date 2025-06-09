import numpy as np
import pandas as pd
import os

def extract_and_save_csv(npz_file, output_csv):
    # Load the .npz file
    data = np.load(npz_file)

    # Extract eigenvalues at (π, π) and Chern numbers
    eigenvalues_pi_pi = data["eigsPipi"]  # Ensure this key exists in your .npz file
    chern_numbers = data["ChernNumbers"]  # Ensure this key exists in your .npz file

    # Create a DataFrame
    df = pd.DataFrame({
        "eigsPipi": eigenvalues_pi_pi,
        "ChernNumber": chern_numbers
    })

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV: {output_csv}")

# Example usage
npz_file = "/Users/eddiedeleu/Downloads/N=1024_Mem/trial_data_94_2025-02-15_20-04-42.npz"  # Replace with the actual .npz file path
output_csv = os.path.splitext(npz_file)[0] + "_chern.csv"  # Save with a related name

extract_and_save_csv(npz_file, output_csv)

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from tqdm import tqdm
import cloudpickle
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

from matplotlib import rc

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern"],
#     "axes.titlesize": 10,
#     "axes.labelsize": 10,
#     "legend.fontsize": 8,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "figure.dpi": 300,
#     "lines.linewidth": 1,
#     "grid.alpha": 0.3,
#     "axes.grid": True
# })

# rc('text.latex', preamble=r'\usepackage{amsmath}')
def load_eigenvalues(folder_path):
    """Load eigenvalues from all .npz files in directory, filtering for SumChernNumbers=1"""
    valid_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    all_eigs = []
    for fname in tqdm(valid_files, desc=f"Loading {os.path.basename(folder_path)}"):
        data = np.load(os.path.join(folder_path, fname))
        if not np.isclose(data['SumChernNumbers'], 1, atol=1e-5):
            continue
        eigs = data['eigsPipi']
        cherns = data['ChernNumbers']

        eigs = eigs[np.isclose(cherns, +1, atol=1e-5)]
        all_eigs.extend(eigs)
    return np.array(all_eigs)

def create_and_save_kde(eigenvalues, save_path):
    kde = gaussian_kde(eigenvalues) #, bw_method=0.1
    with open(save_path, "wb") as f:
        cloudpickle.dump(kde, f)
    return kde

def load_kde_model(load_path):
    with open(load_path, "rb") as f:
        kde = cloudpickle.load(f)
    return kde

def build_idos_interp_from_box1d(kde, energy_grid):
    min_E = energy_grid[0]
    # idos_vals = np.array([kde.integrate_box_1d(min_E, e) for e in energy_grid])
    # idos_vals /= idos_vals[-1]  # Normalize to 1

    idos_vals = Parallel(n_jobs=-1, batch_size=64)(
        delayed(kde.integrate_box_1d)(min_E, E) for E in energy_grid
    )
    idos_vals = np.array(idos_vals)
    idos_vals /= idos_vals[-1]

    return interp1d(energy_grid, idos_vals, bounds_error=False, fill_value=(0.0, 1.0))

def unfold_eigenvalues_in_folder(folder_path, kde, save_dir):
    valid_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    os.makedirs(save_dir, exist_ok=True)

    min_energy = -4
    max_energy = 4
    # idosNorm = quad(kde, min_energy, max_energy)[0]
    # print("IDOS normalization:", idosNorm)

    # def idos_func(E):
    #     return np.array([quad(kde, min_energy, e)[0] for e in np.atleast_1d(E)]) / idosNorm

    idosNorm = kde.integrate_box_1d(min_energy, max_energy)
    print("IDOS normalization:", idosNorm)
    #     def idos_func(E):
        # E = np.atleast_1d(E)
        # return np.array([kde.integrate_box_1d(min_energy, e) for e in E]) / idosNorm

    # Step 2: Define parallel version of idos_func
    # def idos_func(E, n_jobs=-1, batch_size=64):
    #     """
    #     Parallelized computation of IDOS(E) = integrate_box_1d(min_E, E) / IDOS_norm
    #     for arbitrary input array E.
    #     """
    #     E = np.atleast_1d(E)
    #     results = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
    #         delayed(kde.integrate_box_1d)(min_energy, e) for e in E
    #     )
    #     return np.array(results) / idosNorm

    energy_grid = np.linspace(min_energy, max_energy, 5000) #10000
    idos_func = build_idos_interp_from_box1d(kde, energy_grid)
    # plot_kde_and_cdf(energy_grid, kde, idos_func)
    print("0.03 window:", idos_func(-0.03),idos_func(+0.03))
    print("0.1 window:", idos_func(-0.1),idos_func(+0.1))

    for fname in tqdm(valid_files, desc=f"Unfolding {os.path.basename(folder_path)}"):
        data = np.load(os.path.join(folder_path, fname))
        if not np.isclose(data['SumChernNumbers'], 1, atol=1e-5):
            continue

        eigs = data['eigsPipi']
        cherns = data['ChernNumbers']

        ## FILTER FOR RELEVANT CHERNS
        eigs = eigs[np.isclose(cherns, +1, atol=1e-5)]
        cherns = cherns[np.isclose(cherns, +1, atol=1e-5)]

        unfolded = idos_func(eigs) * len(eigs)

        save_path = os.path.join(save_dir, f"unfolded1_{fname}")
        np.savez(save_path, unfolded_eigs=unfolded, original_eigs=eigs,chern=cherns)

def plot_kde_and_cdf(energy_grid, kde, idos, output_path="SaveKDE_CDF_plotParallelized.pdf"):
    """
    Plots both the KDE (PDF) and integrated CDF from the energy grid.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    print("KDE Plotting...")
    # Plot PDF (KDE)
    ax[0].plot(energy_grid, kde(energy_grid), label="KDE PDF")
    ax[0].set_xlabel("Energy")
    ax[0].set_ylabel("Density")
    ax[0].set_title("DOS Estimate (KDE)")
    ax[0].legend()

    print("iDOS Plotting...")
    # Plot CDF
    idos_vals = idos(energy_grid)
    ax[1].plot(energy_grid, idos_vals, label="IDOS (CDF)")
    ax[1].set_xlabel("Energy")
    ax[1].set_ylabel("Cumulative Probability")
    ax[1].set_title("Integrated DOS (CDF)")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Example usage:
if __name__ == "__main__":
    base_folder = "/scratch/network/ed5754/iqheFiles/Data/N=1024_Mem/"
    kde_save_path = "kde_model.cp.pkl"

    # 1. Load and concatenate all eigenvalues
    all_eigs = []
    all_eigs.extend(load_eigenvalues(base_folder))
    all_eigs = np.array(all_eigs)
    all_eigs = np.concatenate((all_eigs, -all_eigs))

    # 2. Create and save KDE
    kde = create_and_save_kde(all_eigs, kde_save_path)
    # kde= load_kde_model("/Users/eddiedeleu/Downloads/kde_model.cp.pkl")

    # 3. Unfold each folder's spectrum and save
    save_dir = "/scratch/network/ed5754/iqheFiles/Data/N=1024_UnfoldedChern1/"
    unfold_eigenvalues_in_folder(base_folder, kde, save_dir)

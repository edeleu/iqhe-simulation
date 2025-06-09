#This file fits and plots the convergence data for Chern number convergence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import os

# Set global matplotlib style configurations
plt.rcParams.update({
    "text.usetex": True,                   # Use LaTeX for text rendering
    "font.family": "serif",                # Use a serif font (e.g., Times New Roman)
    "font.serif": ["Computer Modern"],     # Specify Times New Roman
    "axes.titlesize": 10,                  # Title font size
    "axes.labelsize": 10,                  # Label font size
    "legend.fontsize": 8,                  # Legend font size
    "xtick.labelsize": 8,                  # X-axis tick font size
    "ytick.labelsize": 8,                  # Y-axis tick font size
    # "figure.figsize": (3.5, 2.5),          # Default figure size for single-column plots
    # "figure.dpi": 300,                     # High resolution for publication
    "figure.dpi": 150,                     # High resolution for publication
    "lines.linewidth": 1,                  # Line width
    # "grid.linestyle": "--",                # Dashed grid lines
    "grid.color": "gray",                  # Grid line color
    "grid.alpha": 0.7,                     # Grid line transparency
    "axes.grid": True,                     # Enable grid by default
    "legend.loc": "best",                  # Default legend location
})

# Configure the font rendering with LaTeX for compatibility
rc('text.latex', preamble=r'\usepackage{amsmath}')  # Allows using AMS math symbols

# Example data: Simulated multiple trials for each N
N_values = np.array([8,16, 32, 64, 96, 128, 256, 512])

## FAKE DATA CREATION
# all_trials = {  
#     39: np.random.normal(12, 1.5, 100),  
#     64: np.random.normal(10, 1.2, 100),  
#     96: np.random.normal(8, 1.0, 100),  
#     128: np.random.normal(7, 0.8, 100),  
#     256: np.random.normal(6, 0.5, 100)  
# }

folderPath = "/Users/eddiedeleu/Desktop/convergence"
all_trials = {}
for N in N_values:
    file_name = f"{N}_full.csv" #	chern_{N}_convergenceV3_0.1_results.csv
    file_path = os.path.join(folderPath, file_name)
    df = pd.read_csv(file_path)  # Load CSV
    data_column = df.iloc[1:, 1].astype(float).values  # Extract second column, skipping header
    all_trials[N] = data_column
    print(len(data_column))

# Compute statistics
theta_90th = np.array([np.percentile(all_trials[N], 95) for N in N_values])
theta_99th = np.array([np.percentile(all_trials[N], 99) for N in N_values])
theta_999th = np.array([np.percentile(all_trials[N], 99.9) for N in N_values])

## TEST REGRESSION
from scipy.optimize import curve_fit

# Define power-law function
def power_law(N, A, b):
    return A * N**b

# Fit power law using scipy's curve_fit
popt, pcov = curve_fit(power_law, N_values, theta_99th)

# Extract A and b
A_fit, b_fit = popt
print(f"Fitted parameters: A = {A_fit:.4f}, b = {b_fit:.4f}")

# Generate smooth curve for fitting
N_fit = np.linspace(min(N_values), max(N_values), 100)
theta_fit = power_law(N_fit, A_fit, b_fit)

# Compute fitted values
theta_99th_fit = power_law(N_values, A_fit, b_fit)

# Compute R-squared
SS_res = np.sum((theta_99th - theta_99th_fit) ** 2)  # Residual sum of squares
SS_tot = np.sum((theta_99th - np.mean(theta_99th)) ** 2)  # Total sum of squares
R_squared = 1 - (SS_res / SS_tot)

# Compute RMSE
RMSE = np.sqrt(np.mean((theta_99th - theta_99th_fit) ** 2))

# Print fit quality metrics
print(f"R-squared: {R_squared:.4f}")
print(f"RMSE: {RMSE:.4f}")


### DONE REGRESSING 

# Create the plot
plt.figure(figsize=(8, 5))

# Plot main 99th percentile curve
plt.plot(N_values, theta_99th, 'o-', label="99th Percentile Convergence", color='b')
# plt.plot(N_fit, theta_fit, 'r--', label=r"Fit")
plt.plot(N_fit, theta_fit, 'r--', label=fr"Fit: $\theta = {A_fit:.2f} N^{{{b_fit:.2f}}}$")

# Add shaded regions
plt.fill_between(N_values, theta_90th, theta_99th, alpha=0.3, color='blue', label="95th-99th Percentile Range")
plt.fill_between(N_values, theta_99th, theta_999th, alpha=0.2, color='red', label="99th-99.9th Percentile Range")

# Labels and formatting
plt.xlabel("Number of States (N)", fontsize=14)
plt.ylabel("Converged $\\theta \\times \\theta$ Resolution", fontsize=14)
plt.title("Chern Number Convergence (95th - 99.9th Percentile)", fontsize=16)
plt.xscale("log")
plt.xticks(N_values, labels=N_values)
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.6)

# Show plot
plt.show()

# plt.savefig("ConvergencePlot.pdf", format="pdf", bbox_inches="tight")

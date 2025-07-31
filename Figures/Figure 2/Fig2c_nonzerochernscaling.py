from collections import Counter
import scipy.stats as stats
import os
import numpy as np
import csv
import timeit
# from numba import njit, prange, jit
from timeit import default_timer as timer
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib import rc
from scipy.optimize import curve_fit
from matplotlib.ticker import ScalarFormatter, MaxNLocator

# Set global matplotlib style configurations
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "figure.figsize": (3.4, 3),          # Default figure size for single-column plots
    "lines.linewidth": 1,
    "grid.alpha": 0.3,
    "axes.grid": True
})

# Configure the font rendering with LaTeX for compatibility
rc('text.latex', preamble=r'\usepackage{amsmath}')  # Allows using AMS math symbols

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
## regression stuffs
def power_law(N, A, b):
    # return A * N**b
    return A * N**(1-1/(2*b))

def power_lawV2(N, a, b, y, x):
    return a*(1+b/(N**(y)))* N**(x)

def power_lawV3(N, a, b, x):
    return a*(1+b/np.log(N))* N**(x)

def plotChernRelation():
    dataMeans = [
1.497231496,
2.409899074,
4.138923757,
7.216011783,
10.00586747,
12.59989761,
17.43272707,
21.91971655,
38.13765199,
66.20554142,
114.6200583]

    data = [
0.000480347,
0.0011296,
0.002546886,
0.004811596,
0.007665335,
0.009622746,
0.015306903,
0.019343819,
0.035747716,
0.07609329,
0.1555410
]

    # Example data (replace with actual values from your calculations)
    num_states_values = np.array([8, 16, 32, 64,96, 128,192,256,512,1024,2048])  # N_phi values
    mean_nc_values = np.array(dataMeans)  # Mean N_c
    std_error = np.array(data)  # Standard Error of N_c

    ## LET"S PERFORM OUR REGRESSION

    # Fit power law using scipy's curve_fit
    # popt, pcov = curve_fit(power_law, num_states_values, mean_nc_values)
    popt, pcov = curve_fit(power_law, num_states_values[2:], mean_nc_values[2:],sigma=std_error[2:], absolute_sigma=True)

    # Extract A and b

    A_fit, b_fit = popt
    A_err, b_err = np.sqrt(np.diag(pcov))  # Standard deviations of A and b

    # Print results with uncertainties
    print(f"Fitted parameters: $A = {A_fit:.4f} ± {A_err:.4f}$, $b = {b_fit:.4f} ± {b_err:.4f}$")

    # Generate smooth curve for fitting
    N_fit = np.linspace(min(num_states_values), max(num_states_values), 100)
    NC_FIT = power_law(N_fit, A_fit, b_fit)

    # Compute fitted values
    NC_approx = power_law(num_states_values, A_fit, b_fit)

    # Compute R-squared
    SS_res = np.sum((mean_nc_values - NC_approx) ** 2)  # Residual sum of squares
    print(SS_res)
    SS_tot = np.sum((mean_nc_values - np.mean(mean_nc_values)) ** 2)  # Total sum of squares
    R_squared = 1 - (SS_res / SS_tot)

    # Compute RMSE
    RMSE = np.sqrt(np.mean((mean_nc_values - NC_approx) ** 2))

    # Print fit quality metrics
    print(f"R-squared: {R_squared:.4f}")
    print(f"RMSE: {RMSE:.4f}")

    # Create error bar plot
    # plt.figure()
    fig, ax = plt.subplots()
    # plt.errorbar(num_states_values, mean_nc_values, yerr=std_error, fmt='o', capsize=5, capthick=1.5, elinewidth=1.5, label=r"Mean $N_c$ with Std Error")
    plt.plot(
        num_states_values, mean_nc_values,
        marker='o',
        linestyle='none',          # No lines connecting points
        markersize=4.5,
        markerfacecolor="#A0A0A0",  # Filled APS blue color
        markeredgewidth=.6,
        markeredgecolor="#000000",        # Black outline
        # label=r"Mean $N_{C\ne0}$"
    )
    plt.plot(N_fit, NC_FIT, 'b--', color="#4577f6", linewidth=1.2, label=(r"$N_{C\ne0} = A\cdot N_\phi ^{1-\frac{1}{2\nu}}$"))

    # Aesthetics
    plt.xlabel(r"$N_{\phi}$")
    plt.ylabel(r"$\langle N_{C\ne0} \rangle$") # Mean Number of Nonzero Chern States, 
    # plt.ylabel(r"$\langle N_{C\ne0} \rangle$") # Mean Number of Nonzero Chern States, 

    # plt.title(r"$N_c$ vs $N_{\phi}$", fontsize=16)
    plt.xscale("log")  # Use log scale if necessary
    plt.yscale("log")  # Use log scale if necessary

    plt.xticks([8, 16, 32, 64, 128,256,512,1024,2048], labels=[str(N) for N in [8, 16, 32, 64, 128,256,512,1024,2048]])  # Label x-axis
    # plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.gca().yaxis.set_major_formatter(ScalarFormatter())  # Format Y-axis
    # plt.ticklabel_format(axis='y', style='plain')  # Avoid scientific notation
    plt.yticks([1, 5, 10, 50, 100], labels=["1", "5", "10", "50", "100"])

    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.legend(frameon=False)
    legend = plt.legend(loc="upper left", bbox_to_anchor=(0, 0.94), frameon=False)

    # Add separate textbox for fitted parameters just below the legend
    param_text = (
        rf"$A = {A_fit:.4f} \pm {A_err:.4f}$" + "\n" +
        rf"$\nu = {b_fit:.4f} \pm {b_err:.4f}$" + "\n" +
        rf"$R^2 = {R_squared:.4f}$"
    )
    plt.text(
        0.06, 0.72, param_text,
        transform=ax.transAxes,
        fontsize=9,
        linespacing=1.8,  
        verticalalignment='top')


    plt.tight_layout()
    plt.savefig("Figures/Figure2c.pdf")
    # plt.show()


def plotChernRelationV2():
    # Example data (replace with actual values from your calculations)
    num_states_values = np.array([8, 16, 32, 64,96, 128,192,256,512,1024,2048])  # N_phi values
    mean_nc_values = np.array([1.498584748,2.408245334,4.131289793,7.199676492,10.00063913,12.6123348,17.43278464,21.82487502,37.95166304,66.12270999,115.0033375])  # Mean N_c
    std_error = np.array([0.001263115,0.003328158,0.01497818,0.028245886,0.032039204,0.038400233,0.058835873,0.068562662,0.098667728,0.102322464,0.31544183])  # Standard deviation of N_c

    # num_samples = np.array([528918,191495,19478,5599,1041,4286])
    # sqrt_NS = np.sqrt(num_samples)

    # std_error = std_nc_values / sqrt_NS
    # print(std_error)

    ## LET"S PERFORM OUR REGRESSION

    # Fit power law using scipy's curve_fit
    # popt, pcov = curve_fit(power_law, num_states_values, mean_nc_values)
    popt, pcov = curve_fit(power_lawV2, num_states_values, mean_nc_values,p0=[0.3,2.332,1,1],sigma=std_error, absolute_sigma=True)

    # Extract A and b

    A_fit, b_fit, y_fit, x_fit = popt
    A_err, b_err, y_err, x_err = np.sqrt(np.diag(pcov))  # Standard deviations of A and b

    # Print results with uncertainties
    print(f"Fitted parameters: A = {A_fit:.4f} ± {A_err:.4f}, b = {b_fit:.4f} ± {b_err:.4f}")
    print(f"Fitted parameters: y = {y_fit:.4f} ± {y_err:.4f}, x = {x_fit:.4f} ± {x_err:.4f}")


    # Generate smooth curve for fitting
    N_fit = np.linspace(min(num_states_values), max(num_states_values), 100)
    NC_FIT = power_lawV2(N_fit, A_fit, b_fit,y_fit,x_fit)

    # Compute fitted values
    NC_approx = power_lawV2(num_states_values, A_fit, b_fit,y_fit,x_fit)

    # Compute R-squared
    SS_res = np.sum((mean_nc_values - NC_approx) ** 2)  # Residual sum of squares
    print(SS_res)
    SS_tot = np.sum((mean_nc_values - np.mean(mean_nc_values)) ** 2)  # Total sum of squares
    R_squared = 1 - (SS_res / SS_tot)

    # Compute RMSE
    RMSE = np.sqrt(np.mean((mean_nc_values - NC_approx) ** 2))

    # Print fit quality metrics
    print(f"R-squared: {R_squared:.4f}")
    print(f"RMSE: {RMSE:.4f}")

    # Create error bar plot
    plt.figure(figsize=(7,5))
    # plt.errorbar(num_states_values, mean_nc_values, yerr=std_error, fmt='o', capsize=5, capthick=1.5, elinewidth=1.5, label=r"Mean $N_c$ with Std Error")
    plt.plot(num_states_values, mean_nc_values, 'o', label=r"Mean $N_c$")

    plt.plot(N_fit, NC_FIT, 'r--', label=fr"Fit: $N_C = {A_fit:.4f} N_\phi^{{{b_fit:.4f}}}$")

    # Aesthetics
    plt.xlabel(r"Num States, $N_{\phi}$", fontsize=14)
    plt.ylabel(r"Mean Number of Nonzero Chern States, $\langle N_c \rangle$", fontsize=14)
    plt.title(r"$N_c$ vs $N_{\phi}$", fontsize=16)
    plt.xscale("log")  # Use log scale if necessary
    plt.yscale("log")  # Use log scale if necessary

    plt.xticks(num_states_values, labels=[str(N) for N in num_states_values])  # Label x-axis
    # plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.gca().yaxis.set_major_formatter(ScalarFormatter())  # Format Y-axis
    # plt.ticklabel_format(axis='y', style='plain')  # Avoid scientific notation
    plt.yticks([1, 5, 10, 50, 100], labels=["1", "5", "10", "50", "100"])

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)

    plt.show()

# processEverything()
plotChernRelation()
# plotChernRelationV2()
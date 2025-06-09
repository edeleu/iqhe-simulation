import numpy as np
import matplotlib.pyplot as plt

# For reproducibility
np.random.seed(0)
n_samples = 1000000

# Different Poisson distributions with varying lambda
poisson_params = {
    'Poisson(λ=1)': 1,
    'Poisson(λ=3)': 3,
    'Poisson(λ=5)': 5,
    'Poisson(λ=10)': 10,
    'Poisson(λ=20)': 20,
    'Poisson(λ=100)': 100,
    'Poisson(λ=1000)': 1000
}

# Generate data for each lambda
data = {}
for label, lam in poisson_params.items():
    data[label] = np.random.poisson(lam=lam, size=n_samples)

# Define the Poisson equation as a LaTeX string
equation_pois = r'$P(k)=\frac{\lambda^k e^{-\lambda}}{k!}$'

# Create a figure with rows = number of distributions, columns = 3 (LogX, LogY, Linear)
fig, axes = plt.subplots(nrows=len(data), ncols=3, figsize=(15, 12))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

# Loop over each distribution
for i, (dist_label, samples) in enumerate(data.items()):
    # Column 1: LogX
    # We skip k=0 (since log(0) is undefined), so we only plot the positive k values
    ax_logx = axes[i, 0]
    positive_samples = samples[samples > 0]
    if len(positive_samples) > 0:
        # For discrete data, align bins with integer values
        # We can go from min(positive_samples) to max(...) + 1
        k_min = positive_samples.min()
        k_max = positive_samples.max()
        # Create log-spaced bins from k_min to k_max
        # If k_min == k_max, fallback to a single bin
        if k_min < k_max:
            bins_log = np.logspace(np.log10(k_min), np.log10(k_max), 200)
        else:
            bins_log = [k_min-0.5, k_min+0.5]  # trivial fallback
        ax_logx.hist(positive_samples, bins=bins_log, density=True,
                     color='skyblue', edgecolor='black', alpha=0.7)
        ax_logx.set_xscale('log')
    ax_logx.set_title(f'{dist_label} (LogX)')
    ax_logx.set_xlabel('k (log scale, k>0)')
    ax_logx.set_ylabel('Probability')
    ax_logx.text(0.05, 0.95, equation_pois,
                 transform=ax_logx.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

    # Column 2: LogY
    # Here, we can include k=0. We align bins with integers for Poisson.
    ax_logy = axes[i, 1]
    k_min_full = samples.min()
    k_max_full = samples.max()
    bins_lin = np.arange(k_min_full, k_max_full + 2) - 0.5
    ax_logy.hist(samples, bins=bins_lin, density=True,
                 color='salmon', edgecolor='black', alpha=0.7)
    ax_logy.set_yscale('log')
    ax_logy.set_title(f'{dist_label} (LogY)')
    ax_logy.set_xlabel('k')
    ax_logy.set_ylabel('Probability (log scale)')
    ax_logy.text(0.05, 0.95, equation_pois,
                 transform=ax_logy.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

    # Column 3: Linear
    ax_lin = axes[i, 2]
    ax_lin.hist(samples, bins=bins_lin, density=True,
                color='plum', edgecolor='black', alpha=0.7)
    ax_lin.set_title(f'{dist_label} (Linear)')
    ax_lin.set_xlabel('k')
    ax_lin.set_ylabel('Probability')
    ax_lin.text(0.05, 0.95, equation_pois,
                transform=ax_lin.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

# Save to PDF
plt.savefig('poisson_comparison.pdf')
plt.show()

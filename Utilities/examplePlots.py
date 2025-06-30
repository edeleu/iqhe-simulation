import numpy as np
import matplotlib.pyplot as plt

# Set random seed and number of samples for good statistics
np.random.seed(0)
n_samples = 1000000

# --- Define PDFs for Wigner-Dyson ensembles ---
def pdf_goe(s):
    # Wigner-Dyson surmise for GOE: P(s) = (pi/2)*s*exp(-pi s^2/4)
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)

def pdf_gue(s):
    # Wigner-Dyson surmise for GUE: P(s) = (32/pi^2)*s^2*exp(-4 s^2/pi)
    return (32 / (np.pi**2)) * s**2 * np.exp(-4 * s**2 / np.pi)

def pdf_gse(s):
    # Wigner-Dyson surmise for GSE: 
    # P(s) = (2^18/(3^6*pi^3))*s^4*exp(-64 s^2/(9pi))
    return (2**18 / (3**6 * np.pi**3)) * s**4 * np.exp(-64 * s**2 / (9 * np.pi))

# --- Define PDF for the Wigner semicircle distribution ---
def semicircle_pdf(x, R=2):
    # Returns the semicircle density for |x| <= R, else 0.
    return np.where(np.abs(x) <= R, (2 / (np.pi * R**2)) * np.sqrt(R**2 - x**2), 0)

# --- Simple rejection sampling method ---
def rejection_sample(pdf, x_min, x_max, n_samples, pdf_max):
    samples = []
    while len(samples) < n_samples:
        xs = np.random.uniform(x_min, x_max, n_samples)
        ys = np.random.uniform(0, pdf_max, n_samples)
        accepted = xs[ys < pdf(xs)]
        samples.extend(accepted.tolist())
    return np.array(samples[:n_samples])

# --- Estimate maximum values for rejection sampling ---
x_vals = np.linspace(0, 5, 1000)
pdf_goe_max = np.max(pdf_goe(x_vals))
pdf_gue_max = np.max(pdf_gue(x_vals))
pdf_gse_max = np.max(pdf_gse(x_vals))
# For the semicircle PDF, the maximum occurs at x=0; for R=2, that's 1/pi.
semi_max = 1 / np.pi

# --- Generate sample data for each distribution ---
data = {}
data['Lognormal'] = np.random.lognormal(mean=0, sigma=1, size=n_samples)
data['Exponential'] = np.random.exponential(scale=1.0, size=n_samples)
data['Gaussian'] = np.random.normal(loc=0, scale=1, size=n_samples)
data['Wigner-Dyson GOE'] = rejection_sample(pdf_goe, 0, 5, n_samples, pdf_goe_max)
data['Wigner-Dyson GUE'] = rejection_sample(pdf_gue, 0, 5, n_samples, pdf_gue_max)
data['Wigner-Dyson GSE'] = rejection_sample(pdf_gse, 0, 5, n_samples, pdf_gse_max)
data['Pareto'] = np.random.pareto(3, size=n_samples) + 1
# For the semicircle, the domain is [-2, 2]
data['Wigner Semicircle'] = rejection_sample(lambda x: semicircle_pdf(x, R=2), -2, 2, n_samples, semi_max)

# --- Define LaTeX equations for annotation ---
equations = {
    'Lognormal': r'$f(x)=\frac{1}{x\sigma\sqrt{2\pi}}\exp\left(-\frac{(\ln x-\mu)^2}{2\sigma^2}\right)$',
    'Exponential': r'$f(x)=\lambda e^{-\lambda x}$',
    'Gaussian': r'$f(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$',
    'Wigner-Dyson GOE': r'$P(s)=\frac{\pi}{2}\,s\,\exp\left(-\frac{\pi}{4}s^2\right)$',
    'Wigner-Dyson GUE': r'$P(s)=\frac{32}{\pi^2}\,s^2\,\exp\left(-\frac{4}{\pi}s^2\right)$',
    'Wigner-Dyson GSE': r'$P(s)=\frac{2^{18}}{3^{6}\pi^3}\,s^4\,\exp\left(-\frac{64}{9\pi}s^2\right)$',
    'Pareto': r'$f(x)=\frac{3}{x^4}$, for $x>=1$',
    'Wigner Semicircle': r'$\rho(\lambda)=\frac{2}{\pi R^2}\sqrt{R^2-\lambda^2}$, for $|\lambda|<= R$'
}

# --- Create subplot grid ---
# Now we have 8 distributions and 3 plot types (LogX, LogY, Linear)
fig, axes = plt.subplots(nrows=len(data), ncols=3, figsize=(20, 40))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

# Loop over each distribution
for i, dist in enumerate(data.keys()):
    data_full = data[dist]
    # For plots requiring log scale, use only positive values.
    data_positive = data_full[data_full > 0]
    
    # Use 50 bins for the linear histograms
    bins_lin = 1000

    # For LogX plots, use logarithmically spaced bins (based on positive data)
    if len(data_positive) > 0:
        bins_log = np.logspace(np.log10(data_positive.min()), np.log10(data_positive.max()), 1000)
    else:
        bins_log = bins_lin

    # --- Column 1: LogX plot (x-axis log scale) ---
    ax1 = axes[i, 0]
    ax1.hist(data_positive, bins=bins_log, density=False, alpha=0.7,
             color='skyblue', edgecolor='black')
    ax1.set_xscale('log')
    ax1.set_title(f'{dist} (LogX)')
    ax1.set_xlabel('x (log scale)')
    ax1.set_ylabel('Density')
    ax1.text(0.05, 0.95, equations[dist], transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.6))
    
    # --- Column 2: LogY plot (y-axis log scale) ---
    ax2 = axes[i, 1]
    ax2.hist(data_full, bins=bins_lin, density=True, alpha=0.7,
             color='salmon', edgecolor='black')
    ax2.set_yscale('log')
    ax2.set_title(f'{dist} (LogY)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density (log scale)')
    ax2.text(0.05, 0.95, equations[dist], transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.6))
    
    # --- Column 3: Linear plot (both axes linear) ---
    ax3 = axes[i, 2]
    ax3.hist(data_full, bins=bins_lin, density=True, alpha=0.7,
             color='plum', edgecolor='black')
    ax3.set_title(f'{dist} (Linear)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Density')
    # For Exponential and Lognormal, limit x-axis to 5 in linear plots
    if dist in ['Exponential', 'Lognormal']:
        ax3.set_xlim(0, 5)
    ax3.text(0.05, 0.95, equations[dist], transform=ax3.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.6))

# Save the complete figure as a PDF file
plt.savefig('distribution_plots.pdf')
plt.show()

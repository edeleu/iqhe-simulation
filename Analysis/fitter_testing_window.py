import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from matplotlib import gridspec, rc
import fitter
import scipy.stats as st

# Configure plot settings

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "lines.linewidth": 1,
    "grid.alpha": 0.3,
    "axes.grid": True
})
rc('text.latex', preamble=r'\usepackage{amsmath}')
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def process_trial(data, energy_range):
    eigs = data
    mask = (eigs >= energy_range[0]) & (eigs <= energy_range[1])
    filtered_eigs = eigs[mask]
    separations = np.diff(filtered_eigs) if len(filtered_eigs) > 1 else np.array([])
    return separations

def normalize_separations(separations):
    if len(separations) == 0:
        return separations, 0
    avg_separation = np.mean(separations)
    normalized_seps = separations / avg_separation
    return normalized_seps, avg_separation

# def get_dynamic_bin_count(n_points):
#     if n_points >= 50000:
#         return 500
#     elif n_points >= 10000:
#         return int(100 + (n_points - 10000) * 400 / 40000)
#     elif n_points >= 5000:
#         return int(50 + (n_points - 5000) * 50 / 5000)
#     else:
#         return max(20, int(n_points / 100))
def get_dynamic_bin_count(n_points):
    return 50
    """
    Returns a dynamic number of bins based on the number of data points.
    For n_points ~1,300, this gives ~30 bins.
    For n_points ~1,300,000, this gives ~250 bins (maximum).
    """
    A = 3.25
    alpha = 0.31
    bins = int(np.ceil(A * n_points**alpha))
    return min(bins, 250)

def generate_combined_plots2(eigenvalues, energy_range, all_separations_pos, all_separations_neg, pdf, fit_distributions=True):
    """
    Generates a combined plot with:
      - DOS (top)
      - Linear-scale separation histogram (bottom left)
      - Log-scale separation histogram (bottom right)
    and overlays the top-fit distributions (if fit_distributions=True) on both separation plots.
    
    The legend (positioned to the side) includes the distribution name, fitted parameters, SSE, KS statistic, and p-value.
    
    Args:
        eigenvalues (np.array): All eigenvalues (for the DOS plot).
        energy_range (tuple): Positive energy range (min, max).
        all_separations_pos (np.array): Eigenvalue separations from the positive energy range.
        all_separations_neg (np.array): Eigenvalue separations from the negative energy range.
        pdf (PdfPages): PdfPages object to which plots are saved.
        fit_distributions (bool): If True, perform distribution fitting using Fitter.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import scipy.stats as st
    import fitter

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)  # 2 rows x 2 columns

    # -------------------- Density of States Plot --------------------
    ax_dos = fig.add_subplot(gs[0, :])
    num_bins = 300
    bin_edges = np.linspace(eigenvalues.min(), eigenvalues.max(), num_bins + 1)
    
    # Count eigenvalues in positive and negative regions
    count_pos = np.sum((eigenvalues >= energy_range[0]) & (eigenvalues <= energy_range[1]))
    count_neg = np.sum((eigenvalues >= -energy_range[1]) & (eigenvalues <= -energy_range[0]))
    total_in_ranges = count_pos + count_neg
    percentage_in_ranges = (total_in_ranges / len(eigenvalues)) * 100

    ax_dos.hist(eigenvalues, bins=bin_edges, density=True, alpha=0.7, label="All Eigenvalues")
    ax_dos.axvspan(energy_range[0], energy_range[1], color='green', alpha=0.3, label=f'Positive Range ({count_pos})')
    ax_dos.axvspan(-energy_range[1], -energy_range[0], color='purple', alpha=0.3, label=f'Negative Range ({count_neg})')
    ax_dos.set_xlabel("Energy Eigenvalues")
    ax_dos.set_ylabel("Density of States")
    ax_dos.set_title("Density of States and Regions of Interest")
    ax_dos.text(0.05, 0.95, f'Total in Regions: {total_in_ranges} ({percentage_in_ranges:.2f}%)',
                transform=ax_dos.transAxes, fontsize=10, verticalalignment='top')
    ax_dos.legend(fontsize=8)

    # -------------------- Combined Separations Plot (Linear Scale) --------------------
    ax_sep = fig.add_subplot(gs[1, 0])
    combined_separations = np.concatenate([all_separations_pos, all_separations_neg])
    normalized_seps, avg_sep = normalize_separations(combined_separations)
    n_bins = get_dynamic_bin_count(normalized_seps.size)
    n_points = len(normalized_seps)
    
    ax_sep.hist(normalized_seps, bins=np.linspace(np.min(normalized_seps), np.max(normalized_seps), n_bins),
                density=True, alpha=0.7, label="Combined Separations")
    ax_sep.set_xlabel("Normalized Separation")
    ax_sep.set_ylabel("Density")
    ax_sep.set_title(f'Separation Statistics, Range=[{energy_range[0]:.3f}, {energy_range[1]:.3f}], ⟨s⟩ = {avg_sep:.4f}\nN = {n_points}')
    ax_sep.grid(True, alpha=0.3)

    # -------------------- Distribution Fitting (using Fitter) --------------------
    best_distributions = []
    if fit_distributions and normalized_seps.size > 10:
        try:
            distribution_list = ['gengamma','betaprime','f','chi2','gamma','weibull_max','weibull_min','rice',
                                 'skewnorm','rdist','pearson3','rayleigh','loggamma','nakagami','invweibull',
                                 'maxwell','lognorm','johnsonsu','kstwobign','dweibull','johnsonsb','invgauss',
                                 'invgamma','gumbel_r','genlogistic','genexpon','f','genextreme','gamma','burr12',
                                 'fatiguelife','beta','betaprime','erlang','expon','dgamma','chi2','chi','alpha','poisson']
            f_obj = fitter.Fitter(normalized_seps, distributions=distribution_list, timeout=120)
            f_obj.fit()
            # Sort by sum-square error and take the top 3
            top_results = f_obj.df_errors.sort_values(by='sumsquare_error').head(3)
            best_distributions = top_results.index.tolist()
            # We'll use these fitted parameters for both plots.
        except Exception as e:
            print(f"Fitting failed: {e}")

    # Function to generate a detailed label for a distribution
    def make_label(dist_name):
        try:
            dist = getattr(st, dist_name)
            params = f_obj.fitted_param[dist_name]
            # Get parameter names: shapes (if any) then loc and scale.
            shapes = getattr(st, dist_name).shapes
            if shapes:
                param_list = shapes.split(", ") 
            else:
                param_list = []
            param_list += ["loc", "scale"]
            param_str = ", ".join(f"{n}={p:.6f}" for n, p in zip(param_list, params))
            stats = f_obj.df_errors.loc[dist_name]
            return f"{dist_name}\nSSE: {stats['sumsquare_error']:.6f}, KS: {stats['ks_statistic']:.6f}, p={stats['ks_pvalue']:.3f}\n{param_str}"
        except Exception as e:
            return dist_name

    # Plot fitted distributions on the linear-scale separation plot
    if best_distributions and f_obj is not None:
        x_lin = np.linspace(np.min(normalized_seps), np.max(normalized_seps), 100)
        for dist_name in best_distributions:
            try:
                dist = getattr(st, dist_name)
                params = f_obj.fitted_param[dist_name]
                pdf_values = dist.pdf(x_lin, *params)
                ax_sep.plot(x_lin, pdf_values, label=make_label(dist_name))
            except Exception as e:
                print(f"Plotting {dist_name} on linear scale failed: {e}")
        # ax_sep.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=6)

    # -------------------- Log-Scale Separations Plot --------------------
    ax_logsep = fig.add_subplot(gs[1, 1])
    x_log = np.linspace(np.min(normalized_seps), np.max(normalized_seps), 100)
    ax_logsep.hist(normalized_seps, bins=np.linspace(np.min(normalized_seps), np.max(normalized_seps), n_bins),
                density=True, alpha=0.7)
    ax_logsep.set_yscale('log')
    ax_logsep.set_xlabel("Normalized Separation")
    ax_logsep.set_ylabel("Log Density")
    ax_logsep.set_title(f'Log Scale Separation Statistics, Range=[{energy_range[0]:.3f}, {energy_range[1]:.3f}], ⟨s⟩ = {avg_sep:.4f}\nN = {n_points}')
    ax_logsep.grid(True, which='both', alpha=0.3)
    
    # Plot fitted distributions on the log-scale plot as well
    if best_distributions and f_obj is not None:
        for dist_name in best_distributions:
            try:
                dist = getattr(st, dist_name)
                params = f_obj.fitted_param[dist_name]
                pdf_values = dist.pdf(x_log, *params)
                ax_logsep.plot(x_log, pdf_values, label=make_label(dist_name))
            except Exception as e:
                print(f"Plotting {dist_name} on log scale failed: {e}")
        ax_logsep.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(pdf, format='pdf')
    plt.close(fig)


def generate_combined_plots(eigenvalues, energy_range, all_separations_pos, all_separations_neg, pdf, fit_distributions=True):
    """
    Generates a combined plot with:
      - DOS plot (top row, spanning all columns)
      - Three linear-scale separation plots for 95%, 99.9%, and 100% of the data (middle row)
      - Log-scale separation plot (bottom row, spanning all columns)
    Fitted distributions (if enabled) are overlaid on the linear and log-scale plots. Legends (positioned to the side)
    include distribution name, fitted parameters, SSE, KS statistic, and p-value.
    
    Args:
      eigenvalues (np.array): All eigenvalues for the DOS plot.
      energy_range (tuple): The positive energy range (min, max).
      all_separations_pos (np.array): Eigenvalue separations from the positive range.
      all_separations_neg (np.array): Eigenvalue separations from the negative range.
      pdf (PdfPages): PdfPages object for saving plots.
      fit_distributions (bool): Flag to enable distribution fitting.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import scipy.stats as st
    import fitter

    # Set up a 3-row x 3-column grid:
    fig = plt.figure(figsize=(3.4, 9))
    gs = gridspec.GridSpec(3, 1, figure=fig)
    
    # -------------------- Density of States (DOS) Plot --------------------
    ax_dos = fig.add_subplot(gs[0, :])
    num_bins = 300
    bin_edges = np.linspace(eigenvalues.min(), eigenvalues.max(), num_bins + 1)
    
    count_pos = np.sum((eigenvalues >= energy_range[0]) & (eigenvalues <= energy_range[1]))
    count_neg = np.sum((eigenvalues >= -energy_range[1]) & (eigenvalues <= -energy_range[0]))
    total_in_ranges = count_pos + count_neg
    percentage_in_ranges = (total_in_ranges / len(eigenvalues)) * 100

    ax_dos.hist(eigenvalues, bins=bin_edges, density=True, alpha=0.7, label="All Eigenvalues")
    ax_dos.axvspan(energy_range[0], energy_range[1], color='green', alpha=0.3, label=f'Positive Range ({count_pos}) pts')
    ax_dos.axvspan(-energy_range[1], -energy_range[0], color='purple', alpha=0.3, label=f'Negative Range ({count_neg}) pts')
    ax_dos.set_xlabel("Energy Eigenvalues")
    ax_dos.set_ylabel(r"Normalized DOS $\rho(E)$")
    ax_dos.set_title(f"Total in Regions: {total_in_ranges} pts ({percentage_in_ranges:.2f}\% Total)") #Density of States and Regions of Interest\n
    # ax_dos.text(0.05, 0.95, f'Total in Regions: {total_in_ranges} ({percentage_in_ranges:.2f}\%)',
    #             transform=ax_dos.transAxes, fontsize=10, verticalalignment='top')
    ax_dos.legend(fontsize=6, frameon=False,loc=2)
    
    # -------------------- Prepare Combined Separations Data --------------------
    combined_separations = np.concatenate([all_separations_pos, all_separations_neg])
    normalized_seps, avg_sep = normalize_separations(combined_separations)
    n_points = len(normalized_seps)
    n_bins = get_dynamic_bin_count(normalized_seps.size)
    
    # -------------------- Distribution Fitting (with Fitter) --------------------
    best_distributions = []
    f_obj = None
    if fit_distributions and normalized_seps.size > 10:
        try:
            # distribution_list = ['gengamma','betaprime','f','chi2','gamma','weibull_max','weibull_min',
            #                      'rice','skewnorm','rdist','pearson3','rayleigh','loggamma','nakagami',
            #                      'invweibull','maxwell','lognorm','johnsonsu','kstwobign','dweibull',
            #                      'johnsonsb','invgauss','invgamma','gumbel_r','genlogistic','genexpon',
            #                      'f','genextreme','gamma','burr12','fatiguelife','beta','betaprime',
            #                      'erlang','expon','dgamma','chi2','chi','alpha','poisson']
            distribution_list = ['expon']
            f_obj = fitter.Fitter(normalized_seps, distributions=distribution_list, timeout=120)
            f_obj.fit()
            top_results = f_obj.df_errors.sort_values(by='sumsquare_error').head(3)
            best_distributions = top_results.index.tolist()
        except Exception as e:
            print(f"Fitting failed: {e}")
    
    # -------------------- Linear Separation Plots (Row 1) --------------------
    # Loop over desired percentiles: 95%, 99.9%, and 100% (full range)
    # percentiles = [95, 99.9, 100]
    percentiles = [100]
    for col, percentile in enumerate(percentiles):
        ax_lin = fig.add_subplot(gs[1, col])
        if percentile < 100:
            lower, upper = np.percentile(normalized_seps, [0, percentile])
        else:
            lower, upper = np.min(normalized_seps), np.max(normalized_seps)
        if lower == upper:
            upper += 1e-6
        bins_lin = np.linspace(lower, upper, n_bins)

        # Plot histogram without density normalization
        counts, _ = np.histogram(normalized_seps, bins=bins_lin)

        # Normalize the histogram manually
        bin_widths = np.diff(bins_lin)
        ax_lin.bar(bins_lin[:-1], counts / (len(normalized_seps) * bin_widths), width=bin_widths, alpha=0.7)

        print(np.sum(counts / (len(normalized_seps) * bin_widths) * bin_widths))

        # ax_lin.hist(normalized_seps, bins=bins_lin, density=True, alpha=0.7, range=(lower, upper))
        ax_lin.set_title(f'Range=[{energy_range[0]:.3f}, {energy_range[1]:.3f}], ⟨s⟩ = {avg_sep:.4f}\nN = {n_points}, Bins = {n_bins}')                # ax_lin.set_title(f'Separation Statistics, (0-{percentile}\%)\nRange=[{energy_range[0]:.3f}, {energy_range[1]:.3f}], ⟨s⟩ = {avg_sep:.4f}\nN = {n_points}, Bins = {n_bins}')
        ax_lin.set_xlabel(r"$s/\langle s \rangle$")
        ax_lin.set_ylabel("Density")
        # ax_lin.set_ylim(0,1.7)
        # ax_lin.grid(True, alpha=0.3)
        # Overlay fitted distributions (if available)
        if fit_distributions and f_obj is not None and best_distributions:
            x = np.linspace(lower, upper, 200)
            for i, distr in enumerate(best_distributions):
                color = color_cycle[(i+1) % len(color_cycle)]  
                params = f_obj.fitted_param[distr]
                pdf_vals = getattr(st, distr).pdf(x, *params)
                stats = f_obj.df_errors.loc[distr]
                ks_stat, ks_pvalue, sse = stats["ks_statistic"], stats["ks_pvalue"], stats["sumsquare_error"]

                param_names = getattr(st, distr).shapes
                param_list = param_names.split(", ") if param_names else []
                param_list += ["loc", "scale"]
                param_str = ", ".join(f"{name}={val:.3f}" for name, val in zip(param_list, params))
                ax_lin.plot(x, pdf_vals, 'r--', label=f'{distr} (SSE: {sse:.3f}, KS: {ks_stat:.3f}, p={ks_pvalue:.3f})\n{param_str}',alpha=0.8)
    
    # -------------------- Log-Scale Separation Plot (Row 2, spanning all columns) --------------------
    ax_log = fig.add_subplot(gs[2, :])
    bins_log = np.linspace(np.min(normalized_seps), np.max(normalized_seps), n_bins)
    ax_log.hist(normalized_seps, bins=bins_log, density=True, alpha=0.7)
    ax_log.set_yscale('log')
    ax_log.set_xlabel(r"$s/\langle s \rangle$")
    ax_log.set_ylabel("Log Density")
    ax_log.set_title(f'Range=[{energy_range[0]:.3f}, {energy_range[1]:.3f}]\nN = {n_points}, ⟨s⟩ = {avg_sep:.6f}, Bins = {n_bins}', fontsize=10) #Log Scale Separation Statistics\n
    # ax_log.grid(True, which='both', alpha=0.3)
    if fit_distributions and f_obj is not None and best_distributions:
        x = np.linspace(np.min(normalized_seps), np.max(normalized_seps), 200)
        for i, distr in enumerate(best_distributions):
            color = color_cycle[(i+1) % len(color_cycle)]  
            params = f_obj.fitted_param[distr]
            pdf_vals = getattr(st, distr).pdf(x, *params)
            stats = f_obj.df_errors.loc[distr]
            ks_stat, ks_pvalue, sse = stats["ks_statistic"], stats["ks_pvalue"], stats["sumsquare_error"]

            param_names = getattr(st, distr).shapes
            param_list = param_names.split(", ") if param_names else []
            param_list += ["loc", "scale"]
            param_str = ", ".join(f"{name}={val:.3f}" for name, val in zip(param_list, params))
            # ax_log.plot(x, pdf_vals, color=color, label=f'{distr} (SSE: {sse:.3f}, KS: {ks_stat:.3f}, p={ks_pvalue:.3f})\n{param_str}')
            ax_log.plot(x, pdf_vals, 'r--', label=f'Exponential Fit\n SSE: {sse:.3f}, KS: {ks_stat:.3f}, p={ks_pvalue:.3f}',alpha=0.8)
        # ax_log.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=6)
        ax_log.legend(loc='upper right', frameon=False)
    
    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.tight_layout()
    fig.savefig(pdf, format='pdf')
    plt.close(fig)


def generate_combined_plots_save(eigenvalues, energy_range, all_separations_pos, all_separations_neg, pdf, fit_distributions=True):
    """
    Generates a combined plot with:
      - DOS plot (top row, spanning all columns)
      - Three linear-scale separation plots for 95%, 99.9%, and 100% of the data (middle row)
      - Log-scale separation plot (bottom row, spanning all columns)
    Fitted distributions (if enabled) are overlaid on the linear and log-scale plots. Legends (positioned to the side)
    include distribution name, fitted parameters, SSE, KS statistic, and p-value.
    
    Args:
      eigenvalues (np.array): All eigenvalues for the DOS plot.
      energy_range (tuple): The positive energy range (min, max).
      all_separations_pos (np.array): Eigenvalue separations from the positive range.
      all_separations_neg (np.array): Eigenvalue separations from the negative range.
      pdf (PdfPages): PdfPages object for saving plots.
      fit_distributions (bool): Flag to enable distribution fitting.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import scipy.stats as st
    import fitter

    # Set up a 3-row x 3-column grid:
    fig, ax_dos = plt.subplots(1, 1, figsize=(3.4, 3))    
    # -------------------- Density of States (DOS) Plot --------------------
    # ax_dos = fig.add_subplot(gs[0, :])
    num_bins = 300
    bin_edges = np.linspace(eigenvalues.min(), eigenvalues.max(), num_bins + 1)
    
    count_pos = np.sum((eigenvalues >= energy_range[0]) & (eigenvalues <= energy_range[1]))
    count_neg = np.sum((eigenvalues >= -energy_range[1]) & (eigenvalues <= -energy_range[0]))
    total_in_ranges = count_pos + count_neg
    percentage_in_ranges = (total_in_ranges / len(eigenvalues)) * 100

    ax_dos.hist(eigenvalues, bins=bin_edges, density=True, alpha=0.7, label="All Eigenvalues")
    ax_dos.axvspan(energy_range[0], energy_range[1], color='green', alpha=0.3, label=f'Positive Range ({count_pos}) pts')
    ax_dos.axvspan(-energy_range[1], -energy_range[0], color='purple', alpha=0.3, label=f'Negative Range ({count_neg}) pts')
    ax_dos.set_xlabel("Energy Eigenvalues")
    ax_dos.set_ylabel(r"Normalized DOS $\rho(E)$")
    ax_dos.set_title(f"Total in Regions: {total_in_ranges} pts ({percentage_in_ranges:.2f}\% Total)") #Density of States and Regions of Interest\n
    # ax_dos.text(0.05, 0.95, f'Total in Regions: {total_in_ranges} ({percentage_in_ranges:.2f}\%)',
    #             transform=ax_dos.transAxes, fontsize=10, verticalalignment='top')
    ax_dos.legend(fontsize=6, frameon=False,loc=2)

    plt.tight_layout()
    plt.savefig("DOS_saved1024_to6.pdf")
    
    # -------------------- Prepare Combined Separations Data --------------------
    combined_separations = np.concatenate([all_separations_pos, all_separations_neg])
    normalized_seps, avg_sep = normalize_separations(combined_separations)
    n_points = len(normalized_seps)
    n_bins = get_dynamic_bin_count(normalized_seps.size)
    
    # -------------------- Distribution Fitting (with Fitter) --------------------
    best_distributions = []
    f_obj = None
    if fit_distributions and normalized_seps.size > 10:
        try:
            # distribution_list = ['gengamma','betaprime','f','chi2','gamma','weibull_max','weibull_min',
            #                      'rice','skewnorm','rdist','pearson3','rayleigh','loggamma','nakagami',
            #                      'invweibull','maxwell','lognorm','johnsonsu','kstwobign','dweibull',
            #                      'johnsonsb','invgauss','invgamma','gumbel_r','genlogistic','genexpon',
            #                      'f','genextreme','gamma','burr12','fatiguelife','beta','betaprime',
            #                      'erlang','expon','dgamma','chi2','chi','alpha','poisson']
            distribution_list = ['expon']
            f_obj = fitter.Fitter(normalized_seps, distributions=distribution_list, timeout=120)
            f_obj.fit()
            top_results = f_obj.df_errors.sort_values(by='sumsquare_error').head(3)
            best_distributions = top_results.index.tolist()
        except Exception as e:
            print(f"Fitting failed: {e}")
    
    # -------------------- Linear Separation Plots (Row 1) --------------------
    # Loop over desired percentiles: 95%, 99.9%, and 100% (full range)
    # percentiles = [95, 99.9, 100]
    percentiles = [100]
    for col, percentile in enumerate(percentiles):
        fig, ax_lin = plt.subplots(1, 1, figsize=(3.4, 3))    
        if percentile < 100:
            lower, upper = np.percentile(normalized_seps, [0, percentile])
        else:
            lower, upper = np.min(normalized_seps), np.max(normalized_seps)
        if lower == upper:
            upper += 1e-6
        bins_lin = np.linspace(lower, upper, n_bins)

        # Plot histogram without density normalization
        counts, _ = np.histogram(normalized_seps, bins=bins_lin)

        # Normalize the histogram manually
        bin_widths = np.diff(bins_lin)
        ax_lin.bar(bins_lin[:-1], counts / (len(normalized_seps) * bin_widths), width=bin_widths, alpha=0.7)

        print(np.sum(counts / (len(normalized_seps) * bin_widths) * bin_widths))

        # ax_lin.hist(normalized_seps, bins=bins_lin, density=True, alpha=0.7, range=(lower, upper))
        ax_lin.set_title(f'Range=[{energy_range[0]:.3f}, {energy_range[1]:.3f}], ⟨s⟩ = {avg_sep:.4f}\nN = {n_points}, Bins = {n_bins}')                # ax_lin.set_title(f'Separation Statistics, (0-{percentile}\%)\nRange=[{energy_range[0]:.3f}, {energy_range[1]:.3f}], ⟨s⟩ = {avg_sep:.4f}\nN = {n_points}, Bins = {n_bins}')
        ax_lin.set_xlabel(r"$s/\langle s \rangle$")
        ax_lin.set_ylabel("Density")
        # ax_lin.set_ylim(0,1.7)
        # ax_lin.grid(True, alpha=0.3)
        # Overlay fitted distributions (if available)
        if fit_distributions and f_obj is not None and best_distributions:
            x = np.linspace(lower, upper, 200)
            for i, distr in enumerate(best_distributions):
                color = color_cycle[(i+1) % len(color_cycle)]  
                params = f_obj.fitted_param[distr]
                pdf_vals = getattr(st, distr).pdf(x, *params)
                stats = f_obj.df_errors.loc[distr]
                ks_stat, ks_pvalue, sse = stats["ks_statistic"], stats["ks_pvalue"], stats["sumsquare_error"]

                param_names = getattr(st, distr).shapes
                param_list = param_names.split(", ") if param_names else []
                param_list += ["loc", "scale"]
                param_str = ", ".join(f"{name}={val:.3f}" for name, val in zip(param_list, params))
                ax_lin.plot(x, pdf_vals, 'r--', label=f'{distr} (SSE: {sse:.3f}, KS: {ks_stat:.3f}, p={ks_pvalue:.3f})\n{param_str}',alpha=0.8)
    
    plt.tight_layout()
    plt.savefig("linearsaved1024_to6.pdf")
    # -------------------- Log-Scale Separation Plot (Row 2, spanning all columns) --------------------
    fig, ax_log = plt.subplots(1, 1, figsize=(3.4, 3))    
    bins_log = np.linspace(np.min(normalized_seps), np.max(normalized_seps), n_bins)
    ax_log.hist(normalized_seps, bins=bins_log, density=True, alpha=0.7)
    ax_log.set_yscale('log')
    ax_log.set_xlabel(r"$s/\langle s \rangle$")
    ax_log.set_ylabel("Log Density")
    ax_log.set_title(f'Range=[{energy_range[0]:.3f}, {energy_range[1]:.3f}]\nN = {n_points}, ⟨s⟩ = {avg_sep:.6f}, Bins = {n_bins}', fontsize=10) #Log Scale Separation Statistics\n
    # ax_log.grid(True, which='both', alpha=0.3)
    if fit_distributions and f_obj is not None and best_distributions:
        x = np.linspace(np.min(normalized_seps), np.max(normalized_seps), 200)
        for i, distr in enumerate(best_distributions):
            color = color_cycle[(i+1) % len(color_cycle)]  
            params = f_obj.fitted_param[distr]
            pdf_vals = getattr(st, distr).pdf(x, *params)
            stats = f_obj.df_errors.loc[distr]
            ks_stat, ks_pvalue, sse = stats["ks_statistic"], stats["ks_pvalue"], stats["sumsquare_error"]

            param_names = getattr(st, distr).shapes
            param_list = param_names.split(", ") if param_names else []
            param_list += ["loc", "scale"]
            param_str = ", ".join(f"{name}={val:.3f}" for name, val in zip(param_list, params))
            # ax_log.plot(x, pdf_vals, color=color, label=f'{distr} (SSE: {sse:.3f}, KS: {ks_stat:.3f}, p={ks_pvalue:.3f})\n{param_str}')
            print(f"{param_str}")
            ax_log.plot(x, pdf_vals, 'r--', label=f'Exponential Fit\n SSE: {sse:.3f}, KS: {ks_stat:.3f}, p={ks_pvalue:.3f}',alpha=0.8)
        # ax_log.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=6)
        ax_log.legend(loc='upper right', frameon=False)
    
    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.tight_layout()
    plt.savefig("log_saved1024_to6.pdf")
    # plt.close(fig)

def analyze_eigenvalue_separations(folder_path, initial_range=(4.08, 6), min_data=1000, fit_distributions=False):
    valid_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    print(f"Found {len(valid_files)} .npz files in the folder.")

    with PdfPages('separations_2048_end6_newwx.pdf') as pdf:
        current_range = np.array(initial_range)
        while True:
            # Define symmetric negative range
            current_range_neg = -current_range[::-1]

            print(f"\nProcessing ranges:")
            print(f"Positive: [{current_range[0]:.3f}, {current_range[1]:.3f}]")
            print(f"Negative: [{current_range_neg[0]:.3f}, {current_range_neg[1]:.3f}]")

            all_separations_pos = []
            all_separations_neg = []
            all_eigenvalues = []  # Collect all eigenvalues for the DOS plot

            for i, fname in enumerate(valid_files, 1):
                if i % 10 == 0:
                    print(f"  Processing file {i}/{len(valid_files)}")
                data = np.load(os.path.join(folder_path, fname))
                if not np.isclose(data['SumChernNumbers'], 1, atol=1e-5):
                    continue
                eigenvalues = data['eigsPipi']  # Load eigenvalues from each .npz file
                all_eigenvalues.extend(eigenvalues)  # Append to the list of all eigenvalues

                separations_pos = process_trial(eigenvalues, current_range)
                separations_neg = process_trial(eigenvalues, current_range_neg)

                all_separations_pos.append(separations_pos)
                all_separations_neg.append(separations_neg)

            # Convert to numpy arrays
            combined_pos = np.concatenate(all_separations_pos) if all_separations_pos else np.array([])  # Handle empty arrays
            combined_neg = np.concatenate(all_separations_neg) if all_separations_neg else np.array([])
            all_eigenvalues = np.array(all_eigenvalues)

            min_points = min(len(combined_pos), len(combined_neg))
            print(f"  Total points - Positive: {len(combined_pos):,}, Negative: {len(combined_neg):,}")

            if min_points < min_data:
                print("Stopping due to insufficient data")
                break

            print("  Generating plots...")
            
            # Generate combined plot
            # generate_combined_plots(all_eigenvalues, current_range, combined_pos, combined_neg, pdf, fit_distributions)
            generate_combined_plots_save(all_eigenvalues, current_range, combined_pos, combined_neg, pdf, fit_distributions)
            break

            # Narrow ranges
            range_width = current_range[1] - current_range[0]
            new_width = range_width * 0.97
            current_range = np.array([current_range[1] - new_width, current_range[1]])  # Anchor right bound

    print("\nAnalysis complete. Output saved to combined_separations.pdf")

# Execute analysis (with flag for fitting)
analyze_eigenvalue_separations("/Users/eddiedeleu/Desktop/FinalData/N=1024_MEM", fit_distributions=True)
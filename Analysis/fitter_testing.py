import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from matplotlib import gridspec, rc
import fitter
import scipy.stats as st
from scipy import stats

from fitter import Fitter  # For fitting distributions
from scipy.stats import rv_continuous
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# -- Custom Distributions for Wigner Surmise --
def overlay_gue_curve(ax, s_max=6, num_points=1000, label="GUE", color="green", linestyle="--"):
    s = np.linspace(0, s_max, num_points)
    # p_s = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
    # p_s = np.exp(-1.65*(s-0.6))
    p_s = s*np.exp(-s)

    ax.plot(s, p_s, label=label, color=color, linestyle=linestyle)

# class wigner_gue_gen(rv_continuous):
#     def _pdf(self, s):
#         # Wigner surmise for GUE: P(s) = (32/pi^2) * s^2 * exp(- (4/pi)* s^2)
#         return (32.0 / np.pi**2) * s**2 * np.exp(- (4.0 / np.pi) * s**2)

# class wigner_goe_gen(rv_continuous):
#     def _pdf(self, s):
#         # Wigner surmise for GOE: P(s) = (pi/2) * s * exp(- (pi/4)* s^2)
#         return (np.pi / 2) * s * np.exp(- (np.pi / 4) * s**2)
#     def _fit(self, data, *args, **kwds):
#         # Force a fit: no shape parameters, and fix loc=0, scale=1.
#         return (0, 1)

# wigner_gue = wigner_gue_gen(name='wigner_gue')
# wigner_goe = wigner_goe_gen(name='wigner_goe')
# st.wigner_gue = wigner_gue
# st.wigner_goe = wigner_goe

# Configure plot settings (as before)
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

# Chern filters (as before)
CHERN_FILTERS = {
    r'All Chern': None,
    r'C = $0$ or $+1$': [0, 1],
    r'C = $0$ or $-1$': [0, -1],
    r'C = $0$': [0],
    r'C = $-1$': [-1],
    r'C = $+1$': [1],
    r'$|$C$|$ = 1': [-1, 1],
    r'C $\ne 0$': [-3,-2,-1, 1,2,3]
}

# process_trial, normalize_separations, get_dynamic_bin_count (as before)
def process_trial(data, energy_range, chern_filters):
    eigs = data['eigs']
    chern = data['chern']
    mask = (eigs >= energy_range[0]) & (eigs <= energy_range[1])
    trial_eigs = eigs[mask]
    trial_chern = chern[mask]
    results = {}
    for name, cf in chern_filters.items():
        if cf is None:
            filtered_eigs = trial_eigs
        else:
            c_mask = np.isin(trial_chern, cf)
            filtered_eigs = trial_eigs[c_mask]
        separations = np.diff(filtered_eigs) if len(filtered_eigs) > 1 else np.array([])
        results[name] = separations
    return results

def normalize_separations(separations):
    if len(separations) == 0:
        return separations, 0
    avg_separation = np.mean(separations)
    normalized_seps = separations / avg_separation
    return normalized_seps, avg_separation

# def get_dynamic_bin_count(n_points):
#     # return 100
#     if n_points >= 50000:
#         return 500
#     elif n_points >= 10000:
#         return int(100 + (n_points - 10000) * 400 / 40000)
#     elif n_points >= 5000:
#         return int(50 + (n_points - 5000) * 50 / 5000)
#     else:
#         return max(20, int(n_points / 100))

def get_dynamic_bin_count(n_points):
    """
    Returns a dynamic number of bins based on the number of data points.
    For n_points ~1,300, this gives ~30 bins.
    For n_points ~1,300,000, this gives ~250 bins (maximum).
    """
    A = 3.25
    alpha = 0.31
    bins = int(np.ceil(A * n_points**alpha))
    return min(bins, 250)

# def generate_plotsStable(energy_range, all_separations, pdf, fit_exponential=True):
#     fig = plt.figure(figsize=(20, 5 * len(CHERN_FILTERS)))
#     gs = gridspec.GridSpec(len(CHERN_FILTERS), 4, figure=fig)
#     fig.suptitle(f"Energy Range: [{energy_range[0]:.3f}, {energy_range[1]:.3f}]", y=0.98, fontsize=16)

#     for row, (name, seps) in enumerate(all_separations.items()):
#         if seps.size == 0:
#             continue

#         normalized_seps, avg_sep = normalize_separations(seps)
#         n_bins = get_dynamic_bin_count(normalized_seps.size)

#         # Use Fitter to fit several candidate distributions and retrieve the top five.
#         best_distrs = []
#         f_obj = None
#         if fit_exponential and normalized_seps.size > 10:
#             try:
#                 distribution_list =['gengamma','betaprime','f','chi2','gamma','weibull_max','weibull_min','rice','skewnorm','rdist','pearson3','rayleigh','loggamma','nakagami','invweibull','maxwell','lognorm','johnsonsu','kstwobign','dweibull','johnsonsb','invgauss','invgamma','gumbel_r','genlogistic','genexpon','f','genextreme','gamma','burr12','fatiguelife','beta','betaprime','erlang','expon','dgamma','chi2','chi','alpha','poisson']
#                 f_obj = Fitter(normalized_seps, distributions=distribution_list)
#                 f_obj.fit()
#                 # f.get_best returns a dict sorted by error; take the top five keys
#                 print(f"\nDistribution fitting summary for {name}:")
#                 print(f_obj.summary(Nbest=5))
#                 best_distrs = list(f_obj.get_best(method='sumsquare_error').keys())
#                 print(f"Best distributions for {name}: {best_distrs}")
#             except Exception as e:
#                 print(f"Fitter failed for {name}: {e}")

#         # Loop over the three percentile ranges.
#         for col, percentile in enumerate([95, 99, 99.9]):
#             ax = fig.add_subplot(gs[row, col])
#             lower, upper = np.percentile(normalized_seps, [0, percentile])
#             # Prevent zero-width range
#             if lower == upper:
#                 upper += 1e-6

#             bins = np.linspace(lower, upper, n_bins)
#             ax.hist(normalized_seps, bins=bins, density=True, alpha=0.7, range=(lower, upper))
#             ax.set_title(f'{name} (0-{percentile}%)\n$N = {normalized_seps.size:,}$, Bins = {n_bins}\n$\\langle s \\rangle = {avg_sep:.4f}$', fontsize=10)
#             ax.set_ylabel(r'Density')
#             ax.set_xlabel(r'Normalized Separation $s/\\langle s \\rangle$')
#             ax.grid(True, alpha=0.3)

#             # Plot PDFs for the best five distributions if available.
#             if best_distrs and f_obj is not None:
#                 x = np.linspace(lower, upper, 100)
#                 for distr in best_distrs:
#                     params = f_obj.fitted_param[distr]
#                     pdf_vals = getattr(st, distr).pdf(x, *params)
#                     ax.plot(x, pdf_vals, label=f'{distr} fit')
#                 ax.legend()

#         # Log-scale histogram subplot.
#         ax = fig.add_subplot(gs[row, 3])
#         min_val = np.min(normalized_seps)
#         max_val = np.max(normalized_seps)
#         if min_val == max_val:
#             max_val += 1e-6

#         bins = np.linspace(min_val, max_val, n_bins)
#         ax.hist(normalized_seps, bins=bins, density=True, alpha=0.7)
#         ax.set_yscale('log')
#         ax.set_title(f'{name} - Log Scale (100%)\n$N = {normalized_seps.size:,}$, Bins = {n_bins}\n$\\langle s \\rangle = {avg_sep:.4f}$', fontsize=10)
#         ax.set_ylabel(r'Log Density')
#         ax.set_xlabel(r'Normalized Separation $s/\\langle s \\rangle$')
#         ax.grid(True, alpha=0.3, which='both')

#         if best_distrs and f_obj is not None:
#             x = np.linspace(min_val, max_val, 100)
#             for distr in best_distrs:
#                 params = f_obj.fitted_param[distr]
#                 pdf_vals = getattr(st, distr).pdf(x, *params)
#                 ax.plot(x, pdf_vals, label=f'{distr} fit')
#             ax.legend()

#     plt.tight_layout(rect=[0, 0.03, 1, 0.97])
#     pdf.savefig()
#     plt.close()

# def generate_plots_working(energy_range, all_separations, pdf, fit_exponential=True):
#     fig = plt.figure(figsize=(20, 5 * len(CHERN_FILTERS)))
#     gs = gridspec.GridSpec(len(CHERN_FILTERS), 4, figure=fig)
#     fig.suptitle(f"Energy Range: [{energy_range[0]:.3f}, {energy_range[1]:.3f}]", 
#                  y=0.98, fontsize=16)

#     for row, (name, seps) in enumerate(all_separations.items()):
#         if seps.size == 0:
#             continue

#         normalized_seps, avg_sep = normalize_separations(seps)
#         n_bins = get_dynamic_bin_count(normalized_seps.size)

#         # Fit distributions using Fitter and get the best fit.
#         best_fit_name = None
#         best_fit_params = None
#         best_fit_error = None
#         f_obj = None
#         if fit_exponential and normalized_seps.size > 10:
#             try:
#                 distribution_list = [
#                     'gengamma','betaprime','f','chi2','gamma','weibull_max','weibull_min',
#                     'rice','skewnorm','rdist'
#                 ]
#                 f_obj = Fitter(normalized_seps, distributions=distribution_list)
#                 f_obj.fit()
#                 # Print the summary with the top 5 best fits.
#                 print(f"\nDistribution fitting summary for {name}:")
#                 print(f_obj.summary(Nbest=5,plot=False))
#                 # Get the single best distribution as a dictionary.
#                 best_dict = f_obj.get_best(method='sumsquare_error')
#                 best_fit_name = list(best_dict.keys())[0]
#                 best_fit_params = best_dict[best_fit_name]
#                 # Retrieve SSE from the fitted errors DataFrame.
#                 best_fit_error = f_obj.df_errors.loc[best_fit_name, 'sumsquare_error']
#                 print(f"Best fit for {name}: {best_fit_name} (SSE: {best_fit_error:.5f})")
#             except Exception as e:
#                 print(f"Fitter failed for {name}: {e}")

#         # Loop over the three percentile ranges.
#         for col, percentile in enumerate([95, 99, 99.9]):
#             ax = fig.add_subplot(gs[row, col])
#             lower, upper = np.percentile(normalized_seps, [0, percentile])
#             if lower == upper:
#                 upper += 1e-6  # Prevent zero-width range

#             bins = np.linspace(lower, upper, n_bins)
#             ax.hist(normalized_seps, bins=bins, density=True, alpha=0.7, range=(lower, upper))
#             ax.set_title(f'{name} (0-{percentile}%)\n$N = {normalized_seps.size:,}$, Bins = {n_bins}\n'
#                          f'$\\langle s \\rangle = {avg_sep:.4f}$', fontsize=10)
#             ax.set_ylabel(r'Density')
#             ax.set_xlabel(r'Normalized Separation $s/\langle s \rangle$')
#             ax.grid(True, alpha=0.3)

#             # Plot the best-fit PDF if available.
#             if best_fit_name is not None and f_obj is not None:
#                 x = np.linspace(lower, upper, 100)
#                 # Use keyword arguments because get_best returns a dictionary.
#                 pdf_vals = getattr(st, best_fit_name).pdf(x, **best_fit_params)
#                 params_str = ", ".join(f"{k}={v:.3f}" for k, v in best_fit_params.items())
#                 label_str = f"{best_fit_name} fit\nSSE: {best_fit_error:.3f}\nparams: {params_str}"
#                 ax.plot(x, pdf_vals, 'r-', label=label_str)
#                 ax.legend(fontsize=8)

#         # Log-scale histogram subplot.
#         ax = fig.add_subplot(gs[row, 3])
#         min_val = np.min(normalized_seps)
#         max_val = np.max(normalized_seps)
#         if min_val == max_val:
#             max_val += 1e-6

#         bins = np.linspace(min_val, max_val, n_bins)
#         ax.hist(normalized_seps, bins=bins, density=True, alpha=0.7)
#         ax.set_yscale('log')
#         ax.set_title(f'{name} - Log Scale (100%)\n$N = {normalized_seps.size:,}$, Bins = {n_bins}\n'
#                      f'$\\langle s \\rangle = {avg_sep:.4f}$', fontsize=10)
#         ax.set_ylabel(r'Log Density')
#         ax.set_xlabel(r'Normalized Separation $s/\langle s \rangle$')
#         ax.grid(True, alpha=0.3, which='both')

#         if best_fit_name is not None and f_obj is not None:
#             x = np.linspace(min_val, max_val, 100)
#             pdf_vals = getattr(st, best_fit_name).pdf(x, **best_fit_params)
#             params_str = ", ".join(f"{k}={v:.3f}" for k, v in best_fit_params.items())
#             label_str = f"{best_fit_name} fit\nSSE: {best_fit_error:.3f}\nparams: {params_str}"
#             ax.plot(x, pdf_vals, 'r-', label=label_str)
#             ax.legend(fontsize=8)

#     plt.tight_layout(rect=[0, 0.03, 1, 0.97])
#     pdf.savefig()
#     plt.close()

def generate_plots(energy_range, all_separations, pdf, fit_exponential=True):
    fig = plt.figure(figsize=(20, 5 * len(all_separations)))
    gs = gridspec.GridSpec(len(all_separations), 4, figure=fig)
    fig.suptitle(f"Energy Range: [{energy_range[0]:.3f}, {energy_range[1]:.3f}]", y=0.98, fontsize=16)

    for row, (name, seps) in enumerate(all_separations.items()):
        if seps.size == 0:
            continue

        normalized_seps, avg_sep = normalize_separations(seps)
        n_bins = get_dynamic_bin_count(normalized_seps.size)

        # Use Fitter to fit several candidate distributions and retrieve the top three.
        best_distrs = []
        best_fit_errors = {}
        best_fit_params = {}
        f_obj = None
        if fit_exponential and normalized_seps.size > 10:
            try:
                distribution_list =['gengamma','betaprime','f','chi2','gamma','weibull_max','weibull_min','rice','skewnorm','rdist','pearson3','rayleigh','loggamma','nakagami','invweibull','maxwell','lognorm','johnsonsu','kstwobign','dweibull','johnsonsb','invgauss','invgamma','gumbel_r','genlogistic','genexpon','f','genextreme','gamma','burr12','fatiguelife','beta','betaprime','erlang','expon','dgamma','chi2','chi','alpha','poisson']
                # distribution_list = ['gengamma', 'betaprime', 'f', 'chi2', 'gamma', 'weibull_max']
                
                f_obj = Fitter(normalized_seps, distributions=distribution_list,timeout=60)
                f_obj.fit()

                # Get the top 3 best fits
                best_fits = f_obj.summary(Nbest=3, plot=False)
                best_distrs = best_fits.index.tolist()  # List of top 3 distribution names
                best_fit_errors = best_fits["sumsquare_error"].to_dict()  # SSE values

                # Extract and round parameters for better readability
                for distr in best_distrs:
                    params = f_obj.fitted_param[distr]
                    best_fit_params[distr] = tuple(round(p, 4) for p in params)

                print(f"\nTop 3 fits for {name}:")
                print(best_fits)
            except Exception as e:
                print(f"Fitter failed for {name}: {e}")

        # Loop over the three percentile ranges.
        for col, percentile in enumerate([95, 99, 99.9]):
            ax = fig.add_subplot(gs[row, col])
            lower, upper = np.percentile(normalized_seps, [0, percentile])
            if lower == upper:
                upper += 1e-6  # Avoid zero-width range

            bins = np.linspace(lower, upper, n_bins)
            ax.hist(normalized_seps, bins=bins, density=True, alpha=0.7, range=(lower, upper))
            ax.set_title(f'{name} (0-{percentile}\\%)\n$N = {normalized_seps.size:,}$, Bins = {n_bins}\n$\\langle s \\rangle = {avg_sep:.6f}$', fontsize=10)
            ax.set_ylabel(r'Density')
            ax.set_xlabel(r'Normalized Separation $s/\langle s \rangle$')
            ax.grid(True, alpha=0.3)

            # Plot PDFs for the top 3 best distributions if available.
            if best_distrs and f_obj is not None:
                x = np.linspace(lower, upper, 200)
                for distr in best_distrs:
                    params = f_obj.fitted_param[distr]
                    pdf_vals = getattr(st, distr).pdf(x, *params)
                    sse = best_fit_errors[distr]
                    # param_str = ", ".join(map(str, best_fit_params[distr]))  # Convert tuple to string
                    # ax.plot(x, pdf_vals, label=f'{distr} (SSE: {sse:.3f})\nParams: {param_str}')

                    # Retrieve shape parameter names if they exist
                    param_names = getattr(st, distr).shapes  
                    param_list = param_names.split(", ") if param_names else []  
                    
                    # Always include loc and scale
                    param_list += ["loc", "scale"]  
                    param_str = ", ".join(f"{name}={val:.3f}" for name, val in zip(param_list, params))
                    ax.plot(x, pdf_vals, label=f'{distr} (SSE: {sse:.3f})\n{param_str}')

                # ax.legend(fontsize=8, loc='upper right', frameon=True)  # Smaller legend

        # Log-scale histogram subplot.
        ax = fig.add_subplot(gs[row, 3])
        min_val = np.min(normalized_seps)
        max_val = np.max(normalized_seps)
        if min_val == max_val:
            max_val += 1e-6

        bins = np.linspace(min_val, max_val, n_bins)
        ax.hist(normalized_seps, bins=bins, density=True, alpha=0.7)
        ax.set_yscale('log')
        ax.set_title(f'{name} - Log Scale (100\\%)\n$N = {normalized_seps.size:,}$, Bins = {n_bins}\n$\\langle s \\rangle = {avg_sep:.6f}$', fontsize=10)
        ax.set_ylabel(r'Log Density')
        ax.set_xlabel(r'Normalized Separation $s/\langle s \rangle$')
        ax.grid(True, alpha=0.3, which='both')

        if best_distrs and f_obj is not None:
            x = np.linspace(min_val, max_val, 300)
            for distr in best_distrs:
                params = f_obj.fitted_param[distr]
                pdf_vals = getattr(st, distr).pdf(x, *params)
                sse = best_fit_errors[distr]

                # Retrieve shape parameter names if they exist
                param_names = getattr(st, distr).shapes  
                param_list = param_names.split(", ") if param_names else []  
                
                # Always include loc and scale
                param_list += ["loc", "scale"]  
                param_str = ", ".join(f"{name}={val:.6f}" for name, val in zip(param_list, params))
                ax.plot(x, pdf_vals, label=f'{distr} (SSE: {sse:.6f})\n{param_str}')

            # ax.legend(fontsize=8, loc='upper right', frameon=True)  # Smaller legend
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    pdf.savefig()
    plt.close()



def generate_plotsKS(energy_range, all_separations, pdf, fit_exponential=True):
    # fig = plt.figure(figsize=(20, 5 * len(all_separations)))
    fig = plt.figure(figsize=(15, 5 * len(all_separations)))
    gs = gridspec.GridSpec(len(all_separations), 3, figure=fig)
    fig.suptitle(f"Energy Range: [{energy_range[0]:.3f}, {energy_range[1]:.3f}]", y=0.98, fontsize=16)

    for row, (name, seps) in enumerate(all_separations.items()):
        if seps.size == 0:
            continue

        normalized_seps, avg_sep = normalize_separations(seps)
        n_bins = get_dynamic_bin_count(normalized_seps.size)

        best_distrs = []
        best_fit_stats = {}
        best_fit_params = {}
        f_obj = None

        if fit_exponential and normalized_seps.size > 10:
            try:
                distribution_list =['gengamma','betaprime','f','chi2','gamma','weibull_max','weibull_min','rice','skewnorm','rdist','pearson3','rayleigh','loggamma','nakagami','invweibull','maxwell','lognorm','johnsonsu','kstwobign','dweibull','johnsonsb','invgauss','invgamma','gumbel_r','genlogistic','genexpon','f','genextreme','gamma','burr12','fatiguelife','beta','betaprime','erlang','expon','dgamma','chi2','chi','alpha','poisson']
                f_obj = Fitter(normalized_seps, distributions=distribution_list, timeout=120)
                # 'wigner_gue', 'wigner_goe', 
                f_obj.fit()

                # Select top 3 distributions based on KS statistic
                best_fits = f_obj.df_errors.sort_values(by="ks_statistic").head(3)
                best_distrs = best_fits.index.tolist()

                # Store KS statistics, p-values, and SSE values
                best_fit_stats = best_fits[["ks_statistic", "ks_pvalue", "sumsquare_error"]].to_dict(orient="index")

                # Extract rounded parameters for better readability
                for distr in best_distrs:
                    params = f_obj.fitted_param[distr]
                    best_fit_params[distr] = tuple(round(p, 4) for p in params)

                print(f"\nTop 3 fits for {name}:")
                print(best_fits)
            except Exception as e:
                print(f"Fitter failed for {name}: {e}")

        # Loop over three percentile ranges
        # for col, percentile in enumerate([95, 99, 99.9]):
        for col, percentile in enumerate([99.5]):

            ax = fig.add_subplot(gs[row, col])
            lower, upper = np.percentile(normalized_seps, [0, percentile])
            if lower == upper:
                upper += 1e-6  # Avoid zero-width range

            bins = np.linspace(lower, upper, n_bins)

            # Plot histogram without density normalization
            counts, _ = np.histogram(normalized_seps, bins=bins)
            # Normalize the histogram manually
            bin_widths = np.diff(bins)
            ax.bar(bins[:-1], counts / (len(normalized_seps) * bin_widths), width=bin_widths, alpha=0.7)
            # print(np.sum(counts / (len(normalized_seps) * bin_widths) * bin_widths))

            # ax.hist(normalized_seps, bins=bins, density=True, alpha=0.7, range=(lower, upper))
            ax.set_title(f'{name} (0-{percentile}\%)\n$N = {normalized_seps.size:,}$, Bins = {n_bins}\n$\\langle s \\rangle = {avg_sep:.6f}$', fontsize=10)
            ax.set_ylabel('Density')
            ax.set_xlabel('Normalized Separation $s/\\langle s \\rangle$')
            ax.grid(True, alpha=0.3)

            if best_distrs and f_obj is not None:
                x = np.linspace(lower, upper, 200)
                for i, distr in enumerate(best_distrs):
                    color = color_cycle[(i+1) % len(color_cycle)]  

                    params = f_obj.fitted_param[distr]
                    pdf_vals = getattr(st, distr).pdf(x, *params)
                    stats = best_fit_stats[distr]
                    ks_stat, ks_pvalue, sse = stats["ks_statistic"], stats["ks_pvalue"], stats["sumsquare_error"]

                    param_names = getattr(st, distr).shapes
                    param_list = param_names.split(", ") if param_names else []
                    param_list += ["loc", "scale"]
                    param_str = ", ".join(f"{name}={val:.3f}" for name, val in zip(param_list, params))

                    ax.plot(x, pdf_vals, color=color,label=f'{distr} (SSE: {sse:.3f}, KS: {ks_stat:.3f}, p={ks_pvalue:.3f})\n{param_str}')

        # Log-scale histogram subplot.
        ax = fig.add_subplot(gs[row, 1])
        min_val, max_val = np.min(normalized_seps), np.max(normalized_seps)
        if min_val == max_val:
            max_val += 1e-6

        bins = np.linspace(min_val, max_val, n_bins)
        ax.hist(normalized_seps, bins=bins, density=True, alpha=0.7)
        ax.set_yscale('log')
        ax.set_title(f'{name} - Log Scale (100\%)\n$N = {normalized_seps.size:,}$, Bins = {n_bins}\n$\\langle s \\rangle = {avg_sep:.6f}$', fontsize=10)
        ax.set_ylabel('Log Density')
        ax.set_xlabel('Normalized Separation $s/\\langle s \\rangle$')
        ax.grid(True, alpha=0.3, which='both')

        if best_distrs and f_obj is not None:
            x = np.linspace(min_val, max_val, 300)
            for i, distr in enumerate(best_distrs):
                color = color_cycle[(i+1) % len(color_cycle)]  
                params = f_obj.fitted_param[distr]
                pdf_vals = getattr(st, distr).pdf(x, *params)
                stats = best_fit_stats[distr]
                ks_stat, ks_pvalue, sse = stats["ks_statistic"], stats["ks_pvalue"], stats["sumsquare_error"]

                param_names = getattr(st, distr).shapes
                param_list = param_names.split(", ") if param_names else []
                param_list += ["loc", "scale"]
                param_str = ", ".join(f"{name}={val:.6f}" for name, val in zip(param_list, params))

                ax.plot(x, pdf_vals, color=color, label=f'{distr} (SSE: {sse:.6f}, KS: {ks_stat:.6f}, p={ks_pvalue:.3f})\n{param_str}')

            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        
        # Log-log scale histogram subplot
        ax = fig.add_subplot(gs[row, 2])
        min_val, max_val = np.min(normalized_seps), np.max(normalized_seps)
        if min_val == max_val:
            max_val += 1e-6

        bins = np.logspace(np.log10(min_val), np.log10(max_val), n_bins)
        ax.hist(normalized_seps, bins=bins, density=False, alpha=0.7)
        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.set_title(f'{name} - Log-Log Scale (100%)\n$N = {normalized_seps.size:,}$, Bins = {n_bins}\n$\\langle s \\rangle = {avg_sep:.6f}$', fontsize=10)
        ax.set_ylabel('Log Density')
        ax.set_xlabel('Log Normalized Separation $s/\\langle s \\rangle$')
        ax.grid(True, alpha=0.3, which='both')

        if best_distrs and f_obj is not None:
            x = np.logspace(np.log10(min_val), np.log10(max_val), 300)
            for i, distr in enumerate(best_distrs):
                color = color_cycle[(i+1) % len(color_cycle)]  
                params = f_obj.fitted_param[distr]
                pdf_vals = getattr(st, distr).pdf(x, *params)
                stats = best_fit_stats[distr]
                ks_stat, ks_pvalue, sse = stats["ks_statistic"], stats["ks_pvalue"], stats["sumsquare_error"]

                param_names = getattr(st, distr).shapes
                param_list = param_names.split(", ") if param_names else []
                param_list += ["loc", "scale"]
                param_str = ", ".join(f"{name}={val:.6f}" for name, val in zip(param_list, params))

                ax.loglog(x, pdf_vals, color=color, label=f'{distr} (SSE: {sse:.6f}, KS: {ks_stat:.6f}, p={ks_pvalue:.3f})\n{param_str}')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    pdf.savefig()
    plt.close()


def generate_plots_save(energy_range, all_separations, pdf, fit_exponential=True):
    # fig = plt.figure(figsize=(20, 5 * len(all_separations)))
    # fig.suptitle(f"Energy Range: [{energy_range[0]:.3f}, {energy_range[1]:.3f}]", y=0.98, fontsize=16)

    for row, (name, seps) in enumerate(all_separations.items()):
        if seps.size == 0:
            continue

        normalized_seps, avg_sep = normalize_separations(seps)
        n_bins = get_dynamic_bin_count(normalized_seps.size)

        best_distrs = []
        best_fit_stats = {}
        best_fit_params = {}
        f_obj = None

        if fit_exponential and normalized_seps.size > 10:
            try:
                distribution_list =['gengamma','betaprime','f','chi2','gamma','weibull_max','weibull_min','rice','skewnorm','rdist','pearson3','rayleigh','loggamma','nakagami','invweibull','maxwell','lognorm','johnsonsu','kstwobign','dweibull','johnsonsb','invgauss','invgamma','gumbel_r','genlogistic','genexpon','f','genextreme','gamma','burr12','fatiguelife','beta','betaprime','erlang','expon','dgamma','chi2','chi','alpha','poisson']
                f_obj = Fitter(normalized_seps, distributions=distribution_list, timeout=120)
                # 'wigner_gue', 'wigner_goe', 
                f_obj.fit()

                # Select top 3 distributions based on KS statistic
                best_fits = f_obj.df_errors.sort_values(by="ks_statistic").head(3)
                best_distrs = best_fits.index.tolist()

                # Store KS statistics, p-values, and SSE values
                best_fit_stats = best_fits[["ks_statistic", "ks_pvalue", "sumsquare_error"]].to_dict(orient="index")

                # Extract rounded parameters for better readability
                for distr in best_distrs:
                    params = f_obj.fitted_param[distr]
                    best_fit_params[distr] = tuple(round(p, 4) for p in params)

                print(f"\nTop 3 fits for {name}:")
                print(best_fits)
            except Exception as e:
                print(f"Fitter failed for {name}: {e}")

        # Loop over three percentile ranges
        # for col, percentile in enumerate([95, 99, 99.9]):
        for col, percentile in enumerate([99]):
            fig, ax = plt.subplots(1, 1, figsize=(3.4,2.6))    
            # ax = fig.add_subplot(gs[row, col])
            lower, upper = np.percentile(normalized_seps, [0, percentile])
            if lower == upper:
                upper += 1e-6  # Avoid zero-width range

            bins = np.linspace(lower, upper, n_bins)

            # Plot histogram without density normalization
            counts, _ = np.histogram(normalized_seps, bins=bins)
            # Normalize the histogram manually
            bin_widths = np.diff(bins)
            ax.bar(bins[:-1], counts / (len(normalized_seps) * bin_widths), width=bin_widths, alpha=0.7)
            # print(np.sum(counts / (len(normalized_seps) * bin_widths) * bin_widths))

            # ax.hist(normalized_seps, bins=bins, density=True, alpha=0.7, range=(lower, upper))
            ax.set_title(f'Energy Range: [{energy_range[0]:.3f}, {energy_range[1]:.3f}], $N = {normalized_seps.size:,}$\nBins = {n_bins}, $\\langle s \\rangle = {avg_sep:.6f}$', fontsize=10) #(0-{percentile}\%)\n
            ax.set_ylabel('Density')
            ax.set_xlabel('$s/\\langle s \\rangle$')
            # ax.grid(True, alpha=0.3)
            # overlay_gue_curve(ax)
            ax.text(0.9, 0.9, f'{name}', transform=ax.transAxes, fontsize=14, verticalalignment='top',horizontalalignment="right")
            ax.text(0.9, 0.7, f'(0-{percentile}\%)', transform=ax.transAxes, fontsize=14, verticalalignment='top',horizontalalignment="right")

            plt.tight_layout()
            plt.savefig(f'{name}_linearctr99.pdf')

            if best_distrs and f_obj is not None:
                x = np.linspace(lower, upper, 200)
                for i, distr in enumerate(best_distrs):
                    color = color_cycle[(i+1) % len(color_cycle)]  

                    params = f_obj.fitted_param[distr]
                    pdf_vals = getattr(st, distr).pdf(x, *params)
                    stats = best_fit_stats[distr]
                    ks_stat, ks_pvalue, sse = stats["ks_statistic"], stats["ks_pvalue"], stats["sumsquare_error"]

                    param_names = getattr(st, distr).shapes
                    param_list = param_names.split(", ") if param_names else []
                    param_list += ["loc", "scale"]
                    param_str = ", ".join(f"{name}={val:.3f}" for name, val in zip(param_list, params))

                    ax.plot(x, pdf_vals, color=color,label=f'{distr} (SSE: {sse:.3f}, KS: {ks_stat:.3f}, p={ks_pvalue:.3f})\n{param_str}')

        # Log-scale histogram subplot.
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.6))    
        min_val, max_val = np.min(normalized_seps), np.max(normalized_seps)
        if min_val == max_val:
            max_val += 1e-6

        bins = np.linspace(min_val, max_val, n_bins)
        ax.hist(normalized_seps, bins=bins, density=True, alpha=0.7)
        ax.set_yscale('log')
        ax.set_title(f'Energy Range: [{energy_range[0]:.3f}, {energy_range[1]:.3f}], $N = {normalized_seps.size:,}$\nBins = {n_bins}, $\\langle s \\rangle = {avg_sep:.6f}$', fontsize=10) #(0-{percentile}\%)\n
        ax.set_ylabel('Log Density')
        ax.set_xlabel('$s/\\langle s \\rangle$')
        # ax.grid(True, alpha=0.3, which='both')
        ax.text(0.9, 0.9, f'{name}', transform=ax.transAxes, fontsize=14, verticalalignment='top',horizontalalignment="right")
        overlay_gue_curve(ax)
        ax.set_ylim(1e-4, 1.7)

        plt.tight_layout()
        plt.savefig(f'{name}_logctr.pdf')

        if best_distrs and f_obj is not None:
            x = np.linspace(min_val, max_val, 300)
            for i, distr in enumerate(best_distrs):
                color = color_cycle[(i+1) % len(color_cycle)]  
                params = f_obj.fitted_param[distr]
                pdf_vals = getattr(st, distr).pdf(x, *params)
                stats = best_fit_stats[distr]
                ks_stat, ks_pvalue, sse = stats["ks_statistic"], stats["ks_pvalue"], stats["sumsquare_error"]

                param_names = getattr(st, distr).shapes
                param_list = param_names.split(", ") if param_names else []
                param_list += ["loc", "scale"]
                param_str = ", ".join(f"{name}={val:.6f}" for name, val in zip(param_list, params))

                ax.plot(x, pdf_vals, color=color, label=f'{distr} (SSE: {sse:.6f}, KS: {ks_stat:.6f}, p={ks_pvalue:.3f})\n{param_str}')

            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        
        # # Log-log scale histogram subplot
        # ax = fig.add_subplot(gs[row, 2])
        # min_val, max_val = np.min(normalized_seps), np.max(normalized_seps)
        # if min_val == max_val:
        #     max_val += 1e-6

        # bins = np.logspace(np.log10(min_val), np.log10(max_val), n_bins)
        # ax.hist(normalized_seps, bins=bins, density=False, alpha=0.7)
        # ax.set_yscale('log')
        # ax.set_xscale('log')

        # ax.set_title(f'{name} - Log-Log Scale (100%)\n$N = {normalized_seps.size:,}$, Bins = {n_bins}\n$\\langle s \\rangle = {avg_sep:.6f}$', fontsize=10)
        # ax.set_ylabel('Log Density')
        # ax.set_xlabel('Log Normalized Separation $s/\\langle s \\rangle$')
        # ax.grid(True, alpha=0.3, which='both')

        # if best_distrs and f_obj is not None:
        #     x = np.logspace(np.log10(min_val), np.log10(max_val), 300)
        #     for i, distr in enumerate(best_distrs):
        #         color = color_cycle[(i+1) % len(color_cycle)]  
        #         params = f_obj.fitted_param[distr]
        #         pdf_vals = getattr(st, distr).pdf(x, *params)
        #         stats = best_fit_stats[distr]
        #         ks_stat, ks_pvalue, sse = stats["ks_statistic"], stats["ks_pvalue"], stats["sumsquare_error"]

        #         param_names = getattr(st, distr).shapes
        #         param_list = param_names.split(", ") if param_names else []
        #         param_list += ["loc", "scale"]
        #         param_str = ", ".join(f"{name}={val:.6f}" for name, val in zip(param_list, params))

        #         ax.loglog(x, pdf_vals, color=color, label=f'{distr} (SSE: {sse:.6f}, KS: {ks_stat:.6f}, p={ks_pvalue:.3f})\n{param_str}')
        #     ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

    # plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    # pdf.savefig()
    # plt.close()


# analyze_eigenvalue_separations (as before)
def analyze_eigenvalue_separations(folder_path, initial_range=(-0.3, 0.3), min_data=10000, halt_on_max=False, fit_exponential=False):
    valid_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    print(f"Found {len(valid_files)} .npz files in the folder.")

    with PdfPages('eigenvalue_separations_center_1024_xx.pdf') as pdf:
        current_range = np.array(initial_range)
        iteration = 0

        while True:
            iteration += 1
            print(f"\nIteration {iteration}: Processing energy range [{current_range[0]:.3f}, {current_range[1]:.3f}]")
            all_separations = {name: [] for name in CHERN_FILTERS}

            for i, fname in enumerate(valid_files, 1):
                if i % 10 == 0:
                    print(f"  Processing file {i}/{len(valid_files)}")
                data = np.load(os.path.join(folder_path, fname))
                if not np.isclose(data['SumChernNumbers'], 1, atol=1e-5):
                    continue

                trial_data = {
                    'eigs': data['eigsPipi'],
                    'chern': data['ChernNumbers']
                }
                trial_results = process_trial(trial_data, current_range, CHERN_FILTERS)

                for name, seps in trial_results.items():
                    all_separations[name].append(seps)

            all_separations = {k: np.concatenate(v) for k, v in all_separations.items()}
            data_lengths = [len(v) for v in all_separations.values()]
            min_points = min(data_lengths)
            max_points = max(data_lengths)

            print(f"  Minimum data points across all filters: {min_points}")
            print(f"  Maximum data points across all filters: {max_points}")
            
            if halt_on_max:
                halt_condition = max_points < min_data
                print("  Halting when maximum data points is less than threshold.")
            else:
                halt_condition = min_points < min_data
                print("  Halting when minimum data points is less than threshold.")

            if halt_condition:
                print(f"Stopping at range [{current_range[0]:.3f}, {current_range[1]:.3f}] with minimum {min_points} and maximum {max_points} points")
                break

            print("  Generating plots...")
            # generate_plotsKS(current_range, all_separations, pdf, fit_exponential=fit_exponential)
            generate_plots_save(current_range, all_separations, pdf, fit_exponential=fit_exponential)
            break
            current_range *= 0.85

    print("\nAnalysis complete. PDF file 'eigenvalue_separations_full.pdf' has been generated.")

# Execute analysis (as before)
analyze_eigenvalue_separations(folder_path="/Users/eddiedeleu/Desktop/FinalData/N=2048_MEM",
                               initial_range=(-0.03, 0.03),
                               min_data=6000,
                               halt_on_max=False,
                               fit_exponential=False)
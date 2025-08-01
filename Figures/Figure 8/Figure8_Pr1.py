from pathlib import Path
from typing import List, Tuple, Optional, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from tqdm import tqdm
from scipy import optimize, integrate, stats
import matplotlib.ticker as mtick

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
    "lines.linewidth": 1,
    # "grid.alpha": 0.3,
    # "axes.grid": True,
})
# allow AMS math
rc("text.latex", preamble=r"\usepackage{amsmath}")

# def local_r_values(eigs_sorted: np.ndarray) -> np.ndarray:
#     """Return folded r^{(1)} for a sorted energy list."""
#     diffs = np.diff(eigs_sorted)
#     if diffs.size < 2:
#         return np.empty(0)
#     r = np.minimum(diffs[:-1], diffs[1:]) / np.maximum(diffs[:-1], diffs[1:])
#     return r
def local_r_values(sorted_eigs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return energy centers E_mid and folded local‑r values.

    r_i = min(Δ_{i-1}, Δ_i) / max(Δ_{i-1}, Δ_i) for i ∈ [1, M−2].
    The folded definition maps r ↦ min(r, 1/r) so r ∈ [0, 1].
    """
    M = sorted_eigs.size
    if M < 3:
        return np.empty(0), np.empty(0)

    diffs = np.diff(sorted_eigs)
    r = np.minimum(diffs[:-1], diffs[1:]) / np.maximum(diffs[:-1], diffs[1:])
    E_mid = sorted_eigs[1:-1]
    return E_mid, r

def load_folder_r(folder: Path, cfilter: Optional[int | List[int]] = None, energy_window: Tuple[float, float] = (-0.03, 0.03),
                  symmetrize: bool = True) -> np.ndarray:
    r_all: List[np.ndarray] = []
    files = list(folder.glob("*.npz"))
    for npz in tqdm(files, desc=f"{folder.name}"):
        data = np.load(npz)
        if not np.isclose(data['SumChernNumbers'], 1, atol=1e-5):
            continue
        eigs = data["eigsPipi"]
        cherns = data.get("ChernNumbers")
        if cfilter is not None and cherns is not None:
            mask = np.isin(cherns, cfilter if isinstance(cfilter, (list, tuple)) else [cfilter])
            eigs = eigs[mask]
        if eigs.size < 3:
            continue
        for s in (+1, -1) if symmetrize else (+1,):
            Evals, rvals = local_r_values(np.sort(s * eigs))
            mask = (Evals >= energy_window[0]) & (Evals <= energy_window[1])
            r_values = rvals[mask]
            r_all.append(r_values)
    return np.concatenate(r_all) if r_all else np.empty(0)

def _norm_const(beta: float) -> float:
    """Return normalisation constant C(β) for Atas folded‑r PDF."""
    integrand: Callable[[float], float] = lambda r: (r + r**2) ** beta / (1 + r + r**2) ** (1 + 1.5 * beta)
    Z, _ = integrate.quad(integrand, 0.0, np.inf, epsabs=1e-10, epsrel=1e-8)
    return 1 / Z

def pdf_atas(r: np.ndarray | float, beta: float) -> np.ndarray | float:
    C = _norm_const(beta)
    return C * (r + r**2) ** beta / (1 + r + r**2) ** (1 + 1.5 * beta)

def pdf_folded(r: np.ndarray | float, beta: float) -> np.ndarray | float:
    r = np.asarray(r, dtype=float)
    eps = 1e-12                      # avoid division by zero
    r_safe = np.clip(r, eps, 1)
    return pdf_atas(r_safe, beta) + pdf_atas(1.0 / r_safe, beta) / r_safe**2

def fit_beta(r_vals: np.ndarray) -> float:
    """Maximum‑likelihood fit of β using the Atas PDF."""
    def nll(beta: float) -> float:
        if beta <= 0:
            return np.inf
        pdf = pdf_folded(r_vals, beta)
        return -np.sum(np.log(pdf))
    res = optimize.minimize_scalar(nll, bounds=(0.01, 5.0), method="bounded")
    return res.x

def plot_Pr(ax: plt.Axes, r_vals: np.ndarray, *, bins: int = 65,
            title: str = "", color: str = "steelblue") -> None:
    # Histogram (step‑filled) & mid‑bin points
    counts, edges, _ = ax.hist(r_vals, bins=bins, range=(0, 1), density=True,
                               color=color, alpha=0.3, histtype="stepfilled")
    counts, edges, _ = ax.hist(r_vals, bins=bins, range=(0, 1), density=True,
                               color="k", histtype="step")
    # centers = 0.5 * (edges[:-1] + edges[1:])
    # ax.plot(centers, counts, "o", ms=3, color="k")

    # Fit β and overlay PDF
    beta_hat = fit_beta(r_vals)
    r_grid = np.linspace(0, 1, 400)
    pdf_grid = pdf_folded(r_grid, beta_hat)
    ax.plot(r_grid, pdf_grid, color="red", lw=1.2,alpha=0.8,
            label=rf"Fit $\beta={beta_hat:.3f}$")

    # ── KS statistic & p-value ────────────────────────────────────────
    # 1. tabulate the fitted PDF on a fine grid
    r_grid = np.linspace(0.0, 1.0, 10001)
    pdf_grid = pdf_folded(r_grid, beta_hat)
    cdf_grid = np.cumsum(pdf_grid)
    cdf_grid /= cdf_grid[-1]           # normalise to 1 exactly

    # 2. build an interpolating CDF callable
    cdf_interp = lambda x: np.interp(x, r_grid, cdf_grid,
                                     left=0.0, right=1.0)

    # 3. KS test against that CDF
    ks_D, ks_p = stats.kstest(r_vals, cdf_interp)


    # ── reduced χ²  (observed vs. fitted PDF) ───────────────────────────────
    #   1. observed raw counts in the *same* bins
    obs_counts, edges = np.histogram(r_vals, bins="fd")

    #   2. expected counts from fitted PDF, via bin-wise integration
    def _bin_prob(a, b, beta):
        return integrate.quad(lambda t: pdf_folded(t, beta), a, b,
                              epsabs=1e-9, epsrel=1e-9)[0]
    exp_probs = np.array([_bin_prob(a, b, beta_hat)
                          for a, b in zip(edges[:-1], edges[1:])])
    exp_counts = exp_probs * obs_counts.sum()

    #   3. χ² / dof  (skip bins with exp = 0)
    mask = exp_counts > 0
    chi2  = np.sum((obs_counts[mask] - exp_counts[mask])**2 / exp_counts[mask])
    dof   = mask.sum() - 1        # data bins minus 1 fitted parameter (β)
    chi2_red = chi2 / dof
    
    # ── concise stats block --------------------------------------------------
    txt = (rf"$\chi_\mathrm{{red}}^2={chi2_red:.2f}$"
           "\n" +
           rf"$KS={ks_D:.3g},\ p={ks_p:.3g}$")
    ax.text(0.97, 0.05, txt,
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,linespacing=1.8)

    # Legend for the red fit line
    ax.legend(loc="lower right",
              bbox_to_anchor=(1.0, 0.17),   # x=1.0 (right), y=0.18
              frameon=False, fontsize=9)
    
    # ax.axvline(np.mean(r_vals), color='crimson', linestyle=':', linewidth=1.5,
    #         label=fr"$\langle r \rangle = {np.mean(r_vals):.3f}$")

    ax.set_xlim(0, 1)
    ax.set_xlabel(r"$\tilde{{r}}^{{(1)}}$")
    ax.set_ylabel(r"$\tilde{{P}}(\tilde{{r}}^{{(1)}})$")
    ax.set_title(title)


SCENARIOS = [("All $C$", None), ("$C=0$", 0), ("$C=1$", 1), ("$C=-1$", -1)]

def make_single_panels(base: Path, n_val: int,window: Tuple[float, float] = (-0.03, 0.03)):
    out_dir = Path("Figure 8"); out_dir.mkdir(exist_ok=True)
    sub = base / f"N={n_val}_Mem" if n_val >= 1024 else base / f"N={n_val}"
    for label, cfilt in SCENARIOS:
        r_vals = load_folder_r(sub, cfilter=cfilt, symmetrize=True,energy_window=window)
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        plot_Pr(ax, r_vals, title=label)
        fig.subplots_adjust(left=0.15, right=0.97, bottom=0.14, top=0.91)
        fig.savefig(out_dir / f"r_hist_{label}.pdf")
        plt.close(fig)


def make_three_column(base: Path, n_val: int,window: Tuple[float, float] = (-0.03, 0.03)):
    out_dir = Path("Figure 8"); out_dir.mkdir(exist_ok=True)
    sub = base / f"N={n_val}_Mem" if n_val >= 1024 else base / f"N={n_val}"
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.7), sharey=True,
                             gridspec_kw={"wspace": 0.05})
    for ax, (label, cfilt) in zip(axes, SCENARIOS):
        r_vals = load_folder_r(sub, cfilter=cfilt, symmetrize=True,energy_window=window)
        plot_Pr(ax, r_vals, title=label)
        if ax is not axes[0]:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    # ❶  Use a cleaner formatter --> "0", "0.2", …, "1"  (no trailing ".0")
    for ax in axes:
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.91, wspace=0.05)
    fig.savefig(out_dir / "r_hist_three_column.pdf")
    plt.close(fig)

if __name__ == "__main__":
    BASE = Path("/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData")
    NPHI = 1024
    WINDOW = (-0.03, 0.03)
    make_single_panels(BASE, NPHI, WINDOW)
    make_three_column(BASE, NPHI, WINDOW)

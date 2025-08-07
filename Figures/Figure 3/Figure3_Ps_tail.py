import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm import tqdm
from scipy import stats, integrate
from typing import List, Optional
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ── Matplotlib default styling (APS‑like) ───────────────────────────────────
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
})
rc("text.latex", preamble=r"\usepackage{amsmath}")

# ── constants ───────────────────────────────────────────────────────────────
BASE = Path("/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData")
SUB  = BASE / "N=1024_Mem"     # adjust if path differs
POS_WIN = (3.79,5.75)
NEG_WIN = (-POS_WIN[1], -POS_WIN[0])
OUT_DIR = Path("Figure 3"); OUT_DIR.mkdir(exist_ok=True)

# ── data utilities ──────────────────────────────────────────────────────────

# ── data utilities (per-trial separations)  ────────────────────────────────
def separations_in_window_single(eigs: np.ndarray,
                                 win: tuple[float, float]) -> np.ndarray:
    """Return nearest-neighbour spacings *within the same trial*
       whose energies lie in [Emin,Emax]."""
    mask = (eigs >= win[0]) & (eigs <= win[1])
    e_sorted = np.sort(eigs[mask])
    return np.diff(e_sorted)               # may be empty

def load_tail_separations(folder: Path,
                          pos_win: tuple[float, float],
                          neg_win: tuple[float, float],
                          *,
                          symmetrize: bool = True) -> np.ndarray:
    """Collect spacings from every .npz **independently**, then concat."""
    s_all: list[np.ndarray] = []
    for npz in tqdm(list(folder.glob("*.npz")), desc=f"Loading {folder.name}"):
        data = np.load(npz)
        if not np.isclose(data["SumChernNumbers"], 1, atol=1e-5):
            continue                       # discard invalid trial
        eigs = data["eigsPipi"].astype(float)

        # positive & negative windows per trial
        s_all.append(separations_in_window_single(eigs, pos_win))
        s_all.append(separations_in_window_single(eigs, neg_win))

        # optional sign-symmetrisation of *values* (not needed for spacings)
        if symmetrize:
            s_all.append(separations_in_window_single(-eigs, pos_win))
            s_all.append(separations_in_window_single(-eigs, neg_win))

    return np.concatenate(s_all) if s_all else np.empty(0)

def load_all_eigs(folder: Path) -> np.ndarray:
    eigs: List[np.ndarray] = []
    for npz in tqdm(list(folder.glob("*.npz")), desc=f"Loading {folder.name}"):
        data = np.load(npz)
        if not np.isclose(data['SumChernNumbers'], 1, atol=1e-5):
            continue
        eigs.append(data["eigsPipi"].astype(float))
    return np.concatenate(eigs) if eigs else np.empty(0)


# ── figure generator ────────────────────────────────────────────────────────

def make_tail_Ps_figure() -> None:
    eigs = load_all_eigs(SUB)
    if eigs.size == 0:
        print("[error] no data loaded"); return
    
    # ── percentage of eigenvalues inside the ±window ---------------------------
    count_pos = np.sum((eigs >= POS_WIN[0]) & (eigs <= POS_WIN[1]))
    count_neg = np.sum((eigs >= NEG_WIN[0]) & (eigs <= NEG_WIN[1]))
    total_in_ranges = count_pos + count_neg
    percentage_in_ranges = 100.0 * total_in_ranges / eigs.size

    print(f"Eigenvalues in |E| ∈ [{POS_WIN[0]}, {POS_WIN[1]}]: "
        f"{total_in_ranges} / {eigs.size} = {percentage_in_ranges:.2f}%")

    # spacings taken **within each trial** (symmetrised)
    s_all = load_tail_separations(SUB, POS_WIN, NEG_WIN, symmetrize=False)
    if s_all.size == 0:
        print("[warn] no spacings found in the tail windows"); return
    mean_s = s_all.mean()
    s_tilde = s_all / mean_s

    # main figure layout (inset via GridSpec)
    fig = plt.figure(figsize=(3.4, 3))
    gs  = GridSpec(1,1, figure=fig, left=0.18, right=0.97, bottom=0.14, top=0.95)
    ax  = fig.add_subplot(gs[0])

    # histogram
    bins = 65

    # PCTILE BASED METHOD
    lower, upper = np.percentile(s_tilde, [0, 99.5])
    bins_lin = np.linspace(lower, upper, bins)

    # Plot histogram without density normalization
    counts, _ = np.histogram(s_tilde, bins=bins_lin)

    # Normalize the histogram manually
    bin_widths = np.diff(bins_lin)[0]
    print(np.sum(counts / (len(s_tilde) * bin_widths) * bin_widths))
    counts, edges, _ = ax.hist(s_tilde, bins=bins_lin, weights=np.ones_like(s_tilde) / (len(s_tilde) * bin_widths),
                               color="#d9d9d9", histtype="stepfilled", alpha=0.6)
    ax.hist(s_tilde, bins=bins_lin, weights=np.ones_like(s_tilde) / (len(s_tilde) * bin_widths),
                               color="k", histtype="step")
    
    # ax.hist(s_tilde, bins=bins, density=True,
            # color="k", histtype="step")
    ax.set_yscale("log")
    ax.set_xlim(0, np.max(edges))

    # exponential fit
    lam_hat = 1.
    s_grid = np.linspace(1e-3, np.max(edges), 10000)
    ax.plot(s_grid, lam_hat * np.exp(-lam_hat * s_grid), color="red", lw=1.2,alpha=0.8,
            label=r"$P(\tilde{s})=\exp{(-\tilde{s})}$")

    # stats: KS, p, chi²_r
    cdf_exp = lambda x: 1 - np.exp(-lam_hat * x)
    ks_D, ks_p = stats.kstest(s_tilde, cdf_exp)

    # ── reduced χ²  (observed vs. fitted PDF) ───────────────────────────────
    #   1. observed raw counts in the *same* bins
    obs_counts, edges = np.histogram(s_tilde, bins="fd")

    #   2. expected counts from fitted PDF, via bin-wise integration
    def _bin_prob(a, b, lam_hat):
        return integrate.quad(lambda x: np.exp(-lam_hat * x), a, b,
                              epsabs=1e-9, epsrel=1e-9)[0]
    
    exp_probs = np.array([_bin_prob(a, b, lam_hat)
                          for a, b in zip(edges[:-1], edges[1:])])
    exp_counts = exp_probs * obs_counts.sum()

    #   3. χ² / dof  (skip bins with exp = 0)
    mask = exp_counts > 0
    chi2  = np.sum((obs_counts[mask] - exp_counts[mask])**2 / exp_counts[mask])
    dof   = mask.sum() - 1        # data bins minus 1
    chi2_red = chi2 / dof

    txt = (rf"$\langle s\rangle={mean_s:.4g}$"
           "\n" + rf"$KS={ks_D:.3f},\ p={ks_p:.3g}$"
           "\n" + rf"$\chi_\mathrm{{red}}^2={chi2_red:.2f}$")
    ax.text(0.03, 0.03, txt, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9,linespacing=1.8)
    ax.legend(loc="lower left", bbox_to_anchor=(0,0.22), frameon=False, fontsize=9)

    ax.set_xlabel(r"$\tilde{s}$")
    ax.set_ylabel(r"$P(\tilde{s})$")

    # ── minimal DOS inset: KDE of symmetrized eigs ─────────────────────────────
    # symmetrize eigenvalues
    eigs_sym = np.concatenate([eigs, -eigs])
    # compute KDE
    kde = stats.gaussian_kde(eigs_sym)
    x_dos = np.linspace(eigs_sym.min(), eigs_sym.max(), 1000)
    y_dos = kde(x_dos)

    # create inset and plot
    ax_in = inset_axes(ax, width="40%", height="30%", loc="upper right")
    ax_in.plot(x_dos, y_dos, color="black", lw=1.0)

    # highlight windows
    ax_in.axvspan(*POS_WIN, color="green", alpha=0.4)
    ax_in.axvspan(*NEG_WIN, color="green", alpha=0.4)

    ax_in.set_xlabel(r"$E$")
    ax_in.set_ylabel(r"$\rho(E)$")

    # strip all ticks & spines for minimalism
    # ax_in.set_xticks([]); ax_in.set_yticks([])
    # for spine in ax_in.spines.values():
        # spine.set_visible(False)

    fig.savefig(OUT_DIR / "tail_Ps_N1024.pdf")
    plt.close(fig)

# ── main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_tail_Ps_figure()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
from typing import List, Optional, Tuple
from scipy import stats, integrate, optimize
from tqdm import tqdm

# ── user switches ──────────────────────────────────────────────────────────
DO_FIT       = False      # set False to disable any fitting
OVERLAY_CASE = 3         # 1,2,3 or None  (which fitted PDF to draw)

# ── Matplotlib defaults (APS‑like) ─────────────────────────────────────────
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

# ── constants ─────────────────────────────────────────────────────────────
BASE      = Path("/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData")
SUB       = BASE / "N=1024_Mem"
CENTER_WIN = (-0.03, 0.03)
OUT_DIR   = Path("Figure 3.5"); OUT_DIR.mkdir(exist_ok=True)

CHERN_SCENARIOS: list[tuple[str, Optional[int]]] = [
    ("All", None),
    ("C0", 0),
    ("C1", 1),
]

# ── PDF helpers (from CentralSeparationsFitMLE) ────────────────────────────
from functools import lru_cache

@lru_cache(maxsize=256)
def _norm_coeffs(n: float, y: float) -> Tuple[float, float]:
    from scipy.special import gamma
    gamma1 = gamma((1+n)/y)
    gamma2 = gamma((2+n)/y)
    B = (gamma2/gamma1)**y
    A = (y*B**((1+n)/y))/gamma1
    return A, B

def normalized_pdf(s: np.ndarray, n: float, y: float) -> np.ndarray:
    A, B = _norm_coeffs(n, y)
    return A * s**n * np.exp(-B * s**y)

# negative log‑likelihood ---------------------------------------------------

def nll(params: Tuple[float,...], s_data: np.ndarray, fixed: dict[str,float]|None=None) -> float:
    if fixed is None:
        n, y = params
    elif 'y' in fixed:
        n, y = params[0], fixed['y']
    else:
        n, y = fixed['n'], params[0]
    if n <= -1 or y <= 0:
        return np.inf
    pdfv = normalized_pdf(s_data, n, y)
    if np.any(pdfv <= 0):
        return np.inf
    return -np.sum(np.log(pdfv))

# ── per‑trial separation loader ────────────────────────────────────────────

def separations_trial(eigs: np.ndarray, window: Tuple[float,float]) -> np.ndarray:
    mask = (eigs >= window[0]) & (eigs <= window[1])
    es   = np.sort(eigs[mask])
    return np.diff(es)


def load_center_seps(folder: Path, cfilt: Optional[int]) -> np.ndarray:
    seps: list[np.ndarray] = []
    for npz in tqdm(list(folder.glob("*.npz")), desc=f"{folder.name}/{cfilt}"):
        d = np.load(npz)
        if not np.isclose(d['SumChernNumbers'],1,atol=1e-5):
            continue
        eigs  = d['eigsPipi'].astype(float)
        chern = d['ChernNumbers']
        if cfilt is not None:
            eigs = eigs[np.isin(chern, cfilt)]
        if eigs.size < 3:
            continue
        seps.append(separations_trial(eigs, CENTER_WIN))
        seps.append(separations_trial(-eigs, CENTER_WIN))   # symmetrise
    return np.concatenate(seps) if seps else np.empty(0)

# ── fit dispatcher ---------------------------------------------------------

def perform_fits(s_tilde: np.ndarray):
    fits = []
    # Case 1: fit n, fix y=2
    res1 = optimize.minimize_scalar(lambda n: nll([n], s_tilde, fixed={'y':2}),
                                    bounds=(0.2,5), method='bounded')
    fits.append( (res1.x, 2.0) )
    # Case 2: fit y, fix n=2
    res2 = optimize.minimize_scalar(lambda y: nll([y], s_tilde, fixed={'n':2}),
                                    bounds=(0.2,5), method='bounded')
    fits.append( (2.0, res2.x) )
    # Case 3: fit n & y
    res3 = optimize.minimize(lambda p: nll(p, s_tilde), x0=[2.0,2.0],
                             bounds=[(0.1,8),(0.2,6)], method='Powell')
    fits.append( tuple(res3.x) )
    return fits

# ── stats helpers ----------------------------------------------------------

def ks_stat(s: np.ndarray, n: float, y: float) -> Tuple[float,float]:
    # build CDF via numerical integration grid
    grid = np.linspace(0, max(s)*1.1, 4000)
    cdf_grid = np.cumsum(normalized_pdf(grid, n, y))
    cdf_grid /= cdf_grid[-1]
    cdf_interp = lambda x: np.interp(x, grid, cdf_grid, left=0, right=1)
    return stats.kstest(s, cdf_interp)

def chi2_red(s: np.ndarray, n: float, y: float):
    counts, edges = np.histogram(s, bins="fd")
    total = counts.sum()
    probs = np.array([integrate.quad(lambda t: normalized_pdf(t,n,y), edges[i], edges[i+1])[0]
                      for i in range(len(edges)-1)])
    exp = probs*total
    mask = exp>=5
    if mask.sum()<3:
        return np.nan
    chi2 = np.sum((counts[mask]-exp[mask])**2/exp[mask])
    dof = mask.sum()-2
    return chi2/dof


# ── single‑panel plot ------------------------------------------------------

def panel_plot(ax: plt.Axes, s: np.ndarray, scale: str, setLabel,
               fit_params: Tuple[float,float]|None=None):
    
    if scale=="loglog":
        # log‑spaced bins
        bins = np.logspace(np.log10(min(s)*0.95), np.log10(max(s)*1.05), 65)
        counts, edges, _ = ax.hist(s, bins=bins, density=True,
                                   color="#d9d9d9", histtype="stepfilled", alpha=0.6)
        ax.hist(s, bins=bins, density=True, color="k", histtype="step")
        ax.set_xscale("log"); ax.set_yscale("log")
    else:
        counts, edges, _ = ax.hist(s, bins=65, density=True,
                                   color="#d9d9d9", histtype="stepfilled", alpha=0.6)
        ax.hist(s, bins=65, density=True, color="k", histtype="step")
        if scale=="semilogy":
            ax.set_yscale("log")

    if fit_params is not None:
        n,y = fit_params
        grid = np.linspace(0, edges[-1], 600)
        ax.plot(grid, normalized_pdf(grid,n,y), color="crimson", lw=1.1,
                label=rf"fit $n={n:.2f},y={y:.2f}$")
        ax.legend(frameon=False, fontsize=6, loc="upper right")

    if setLabel:
        ax.set_xlabel(r"$\tilde{s}$")

    if scale=="linear":
        ax.set_ylabel(r"$P(\tilde{s})$")
        ax.set_xlim(0,np.max(edges)*1.05)
        ax.set_ylim(0,1.01)

    if scale=="semilogy":
        ax.set_yscale("log")
        ax.set_xlim(0,np.max(edges)*1.02)

    if scale=="loglog":
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.75*np.min(edges),np.max(edges)*1.2)


## NOTE: CONTINUE WORKING ON Y-SCALE ADJUSTMENTS and LEFT HAND SPACE / LABEL PLACEMENT
# ── main figure ------------------------------------------------------------

def make_figure():
    # gather separations per scenario once
    sep_dict = {lab: load_center_seps(SUB, cf) for lab,cf in CHERN_SCENARIOS}
    mean_dict = {k: v.mean() for k,v in sep_dict.items() if v.size}
    tilde_dict = {k: v/mean_dict[k] for k,v in sep_dict.items() if v.size}

    # optional fits
    fit_dict: dict[str, list[Tuple[float,float]]] = {}
    if DO_FIT:
        for k, s in tilde_dict.items():
            fit_dict[k] = perform_fits(s)

    # master grid figure ----------------------------------------------------
    fig_h = 4.2   # balanced height
    fig = plt.figure(figsize=(6.8, fig_h))
    gs = GridSpec(len(CHERN_SCENARIOS), 3, figure=fig,
                  left=0.1, right=0.98, bottom=0.1, top=0.93,
                  wspace=0.24, hspace=0.23)

    scales = ["linear", "semilogy", "loglog"]
    # column headers
    for j, sc in enumerate(scales):
        ax_head = fig.add_subplot(gs[0, j])
        ax_head.set_title({"linear":r"\textbf{Linear}","semilogy":r"\textbf{Semilog‑y}","loglog":r"\textbf{Log‑log}"}[sc])
        ax_head.axis("off")

    # plot panels
    for i,(lab,_) in enumerate(CHERN_SCENARIOS):
        s = tilde_dict.get(lab, np.empty(0))
        fits = fit_dict.get(lab) if DO_FIT else None
        psel = fits[OVERLAY_CASE-1] if (fits and OVERLAY_CASE) else None
        
        for j,sc in enumerate(scales):
            ax = fig.add_subplot(gs[i, j])
            panel_plot(ax, s, sc, (i == 2), psel)
    
            if j==0:     # row label
                ax.annotate({"All": r"\textbf{All }\,$\mathbf{C}$",
                            "C0" : r"$\mathbf{C = 0}$",
                            "C1" : r"$\mathbf{C = +1}$"}[lab],
                            xy=(-0.37, 0.5), xycoords="axes fraction",
                            ha="center", va="center",
                            fontsize=10, rotation=90)

    fig.savefig(OUT_DIR/"central_Ps_grid.pdf")
    plt.close(fig)

    # save single‑panel PDFs -------------------------------------------------
    for lab,_ in CHERN_SCENARIOS:
        s = tilde_dict.get(lab, np.empty(0))
        fits = fit_dict.get(lab) if DO_FIT else None
        psel = fits[OVERLAY_CASE-1] if (fits and OVERLAY_CASE) else None
        for sc in scales:
            f,ax = plt.subplots(figsize=(3.4,3))
            panel_plot(ax,s,sc,psel)
            f.subplots_adjust(left=0.25,right=0.97,bottom=0.14,top=0.93)
            f.savefig(OUT_DIR / f"Ps_{lab}_{sc}.pdf")
            plt.close(f)

if __name__ == "__main__":
    make_figure()
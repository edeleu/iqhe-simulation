import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.axes import Axes
from tqdm import tqdm
import cmasher as cmr

###############################################################################
# ── Plot & LaTeX defaults ────────────────────────────────────────────────────
###############################################################################
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
    "grid.alpha": 0.3,
    "axes.grid": True,
})
# allow AMS math
rc("text.latex", preamble=r"\usepackage{amsmath}")

###############################################################################
# ── Local‑r statistic ────────────────────────────────────────────────────────
###############################################################################

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

###############################################################################
# ── Helpers ──────────────────────────────────────────────────────────────────
###############################################################################

def compute_quantile_bin_edges(E: np.ndarray, *, tail_frac: float = 0.0025,
                               n_center: int = 35) -> np.ndarray:
    """Return non‑uniform bin edges following the quantile‑binning scheme."""
    if E.size < 2:
        return np.array([])
    E_sorted = np.sort(E)
    q_min, q_max = np.quantile(E_sorted, [tail_frac, 1 - tail_frac])
    center_qs = np.linspace(tail_frac, 1 - tail_frac, n_center + 1)
    center_edges = np.quantile(E_sorted, center_qs)
    edges = np.concatenate([[E_sorted[0]], [q_min], center_edges[1:-1],
                            [q_max, E_sorted[-1]]])
    return edges

###############################################################################
# ── Data loading ─────────────────────────────────────────────────────────────
###############################################################################

def filter_by_chern(eigs: np.ndarray, cherns: Optional[np.ndarray],
                    cfilter: Optional[int | List[int]]) -> np.ndarray:
    if cfilter is None or cherns is None:
        return eigs
    if isinstance(cfilter, int):
        cfilter = [cfilter]
    mask = np.isin(cherns, cfilter)
    return eigs[mask]


def gather_E_r(folder: Path, *, cfilter: Optional[int | List[int]] = None,
               symmetrize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Collect all (E, r) for .npz files in *folder* with optional C‑filter."""
    Es, rs = [], []
    for npz_path in tqdm(list(folder.glob("*.npz")), desc=f"Loading {folder.name}"):
        data = np.load(npz_path)
        eigs = data["eigsPipi"]
        cherns = data.get("ChernNumbers")
        eigs_filt = filter_by_chern(eigs, cherns, cfilter)
        if eigs_filt.size < 3:
            continue
        for sign in (+1, -1) if symmetrize else (+1,):
            eigs_sorted = np.sort(sign * eigs_filt)
            E_mid, r = local_r_values(eigs_sorted)
            Es.append(E_mid)
            rs.append(r)
    if not Es:
        return np.empty(0), np.empty(0)
    return np.concatenate(Es), np.concatenate(rs)

###############################################################################
# ── Plotting utilities ───────────────────────────────────────────────────────
###############################################################################

def add_quantile_trend(E: np.ndarray, r: np.ndarray, ax: Axes,
                       *, tail_frac: float = 0.0025, n_center: int = 35,
                       color: str = "white") -> None:
    edges = compute_quantile_bin_edges(E, tail_frac=tail_frac, n_center=n_center)
    centers = 0.5 * (edges[:-1] + edges[1:])
    digit = np.digitize(E, edges) - 1
    r_mean = np.array([np.mean(r[digit == i]) if np.any(digit == i) else np.nan
                       for i in range(len(centers))])
    ax.plot(centers, r_mean, color=color, lw=1.1, marker=".",
        linestyle='-',          # No lines connecting points
        markersize=3.2,
        markerfacecolor="#FFFFFF",  # Filled APS blue color
        markeredgewidth=.32,
        markeredgecolor="#848484",        # Black outline
        ) #label="Quantile Mean"

    # ── annotate centre + near-edge bins  ────────────────────────────────────
    idxs = [1, len(centers) // 2, len(centers) - 2]   # left, centre, right
    span = centers[-1] - centers[0]                   # x-range for offsets

    left_idx, mid_idx, right_idx = idxs
    for idx in idxs:
        if np.isnan(r_mean[idx]):
            continue
        x_pt, y_pt = centers[idx], r_mean[idx]
        label = rf"$\langle\tilde{{r}}^{{(1)}}\rangle={y_pt:.4f}$"

        if idx == mid_idx:
            # centre bin: place label slightly to the right, no arrow
            # ax.text(x_pt + 0.02 * span, y_pt,
            #         label, color='white', fontsize=6,
            #         ha='left', va='center', zorder=6)
            ax.annotate(label,
                        xy=(x_pt, y_pt), xycoords='data',
                        xytext=(0, 14), textcoords='offset points',
                        color='white', fontsize=6,
                        ha='center', va='bottom',
                        arrowprops=dict(arrowstyle='->',
                                        color="#FFFFFF", lw=0.8,
                                        shrinkA=0., shrinkB=0.5),zorder=5)

        else:
            # left & right bins: label above with a white arrow pointing down
            x_off = -18 if idx == left_idx else 18   # pixel offset left / right
            ax.annotate(label,
                        xy=(x_pt, y_pt), xycoords='data',
                        xytext=(0, -12), textcoords='offset points',
                        color='white', fontsize=6,
                        ha='center', va='top',
                        arrowprops=dict(arrowstyle='->',
                                        color="#FFFFFF", lw=0.8,
                                        shrinkA=0., shrinkB=0.5),zorder=6)


    # # ── annotate centre + near-edge bins ─────────────────────────────────────
    # idxs = [1, len(centers) // 2, len(centers) - 2]   # 2nd, centre, 2nd-last
    # for idx in idxs:
    #     if np.isnan(r_mean[idx]):          # skip empty bins
    #         continue
    #     ax.text(centers[idx], r_mean[idx] + 0.025,
    #             rf"$\langle\tilde{{r}}^{{(1)}}\rangle={r_mean[idx]:.4f}$",
    #             ha="center", va="bottom", color="white", fontsize=6, zorder=6)


def hexbin_panel(E: np.ndarray, r: np.ndarray, ax: Axes,
                 *, gridsize: int = 100, add_cbar: str = "right") -> None:
    hb = ax.hexbin(E, r, gridsize=gridsize,
                   extent=[E.min(), E.max(), 0, 1],
                   cmap=cmr.get_sub_cmap(cmr.torch, 0.0, 0.92),linewidths=0.05)
    if add_cbar == "right":
        cb = ax.figure.colorbar(hb, ax=ax)
        cb.set_label("counts")
    elif add_cbar == "below":
        cb = ax.figure.colorbar(hb, ax=ax, orientation="horizontal",
                                pad=0.2, fraction=0.08)
        cb.set_label("counts")
    add_quantile_trend(E, r, ax)
    # Reference lines
    ax.axhline(0.60266, color="#2FC5E0", linestyle="--", label="GUE",lw=0.9)
    ax.axhline(0.3863, color="#01AE58", linestyle="--", label="Poisson",lw=0.9)
    ax.set_xlabel(r"$E$")
    ax.set_ylabel(r"$\tilde{r}^{(1)}$")
    ax.set_ylim(0, 1)
    ax.set_xlim(0.95*E.min(), 0.95*E.max())
    ax.legend(loc="upper right", fontsize=6, frameon=False,labelcolor="#FFFFFF")

###############################################################################
# ── Figure generation ───────────────────────────────────────────────────────
###############################################################################

CHERN_SCENARIOS = [
    ("All $C$",  None),   # label, cfilter
    ("$C=0$",   0),
    ("$C=1$",   1),
]


def figure_single_panels(base_folder: Path, n_val: int) -> None:
    out_dir = Path("Figure 7"); out_dir.mkdir(exist_ok=True)
    subdir = base_folder / f"N={n_val}_Mem" if n_val >= 1024 else base_folder / f"N={n_val}"
    for label, cfilt in CHERN_SCENARIOS:
        E, r = gather_E_r(subdir, cfilter=cfilt, symmetrize=True)
        if E.size == 0:
            print(f"[warn] no data for {label} in {subdir}")
            continue
        fig, ax = plt.subplots(figsize=(3.4, 3))
        hexbin_panel(E, r, ax, add_cbar="right")
        ax.set_title(label)
        fig.tight_layout()
        fig.savefig(out_dir / f"r_vs_E_{label}.pdf")
        plt.close(fig)


def figure_three_column(base_folder: Path, n_val: int) -> None:
    out_dir = Path("Figure 7"); out_dir.mkdir(exist_ok=True)
    subdir = base_folder / f"N={n_val}_Mem" if n_val >= 1024 else base_folder / f"N={n_val}"
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 3))  # keep 3" tall
    for ax, (label, cfilt) in zip(axes, CHERN_SCENARIOS):
        E, r = gather_E_r(subdir, cfilter=cfilt, symmetrize=True)
        if E.size == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_axis_off()
            continue
        hexbin_panel(E, r, ax, add_cbar="below")
        ax.set_title(label)
    fig.tight_layout()
    fig.savefig(out_dir / "r_vs_E_three_column.pdf")
    plt.close(fig)

###############################################################################
# ── CLI entry ───────────────────────────────────────────────────────────────
###############################################################################

if __name__ == "__main__":
    BASE_PATH = Path("/scratch/gpfs/ed5754/iqheFiles/Full_Dataset/FinalData")
    N_VALUE = 1024  # adjust as needed

    figure_single_panels(BASE_PATH, N_VALUE)
    figure_three_column(BASE_PATH, N_VALUE)

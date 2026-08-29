"""Publication-quality figures produced during and after the GP pipeline.

Figures produced here:
  M1_gp_uncertainty_surface.png   [REPORT §3 Methods]
      2D PCA projection of one representative fold showing GP posterior
      variance as a heatmap.  Training defectives cluster in low-variance
      regions; the novel held-out type lands in high-variance territory.

  M2_pipeline_diagram.png         [REPORT §3 Methods]
      ASCII-style pipeline schematic as a matplotlib figure.

  R_pca_dim_sweep.png             [APPENDIX]
      AUROC (defect-only) vs PCA dimension d ∈ {8,12,16} across all folds.

  R_kernel_selection.png          [REPORT §4 Optimisation]
      Frequency each of the 6 kernels was selected as best by LML.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from .config import FIGURES_DIR, GP_RESULTS_DIR, EVAL_DIR

log = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# M1 — GP Uncertainty Surface (2D visualization GP)
# ---------------------------------------------------------------------------

def plot_gp_uncertainty_surface(
    fold: dict[str, Any],
    save_path: Optional[Path] = None,
    device: Optional[str] = None,
    n_grid: int = 120,
    pca_dim: int = 16,
) -> Path:
    """
    2D PCA visualization of GP posterior variance for one representative fold.

    A 2D PCA is fitted on training defectives (for visualization only).
    A separate 2D GP is fitted in this 2D space to produce the variance heatmap.
    The actual LODTO model uses d=16; this figure illustrates the geometric intuition.
    """
    from .embeddings import load_embeddings
    from .pca_transform import fit_fold_pca
    from .gp_model import fit_gp, predict_gp

    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "M1_gp_uncertainty_surface.png"

    # ── load embeddings ──────────────────────────────────────────────────
    X_train = load_embeddings(fold["train_defective"])
    X_norm  = load_embeddings(fold["val_normal"][:min(30, len(fold["val_normal"]))])
    X_novel = load_embeddings(fold["test_defective"])
    y_train = np.log(np.array(fold["train_severity"], dtype=np.float64))

    # ── 2D PCA for visualization (fitted on training defectives only) ─────
    fpca2 = fit_fold_pca(X_train, n_components=2)
    Z_train = fpca2.transform(X_train)
    Z_norm  = fpca2.transform(X_norm)
    Z_novel = fpca2.transform(X_novel)

    # ── 2D visualization GP ───────────────────────────────────────────────
    log.info("Fitting 2D visualization GP…")
    fitted = fit_gp(
        Z_train.astype(np.float64), y_train,
        device=device,
        n_restarts=5,
        max_iter=200,
        kernels=["rbf", "mat52"],   # fast subset for the visualization GP
    )

    # ── variance on a meshgrid ────────────────────────────────────────────
    margin = 0.5
    x1_min, x1_max = Z_train[:, 0].min() - margin, Z_train[:, 0].max() + margin
    x2_min, x2_max = Z_train[:, 1].min() - margin, Z_train[:, 1].max() + margin
    # extend to include novel points
    x1_min = min(x1_min, Z_novel[:, 0].min() - margin)
    x1_max = max(x1_max, Z_novel[:, 0].max() + margin)
    x2_min = min(x2_min, Z_novel[:, 1].min() - margin)
    x2_max = max(x2_max, Z_novel[:, 1].max() + margin)

    xx, yy = np.meshgrid(
        np.linspace(x1_min, x1_max, n_grid),
        np.linspace(x2_min, x2_max, n_grid),
    )
    mesh = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
    _, var_mesh = predict_gp(mesh, fitted)
    Var = var_mesh.reshape(xx.shape)

    # ── figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 5.5))
    gs  = GridSpec(1, 2, width_ratios=[5, 1], wspace=0.05)
    ax  = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    im = ax.contourf(xx, yy, Var, levels=30, cmap="YlOrRd", alpha=0.85)
    ax.contour(xx, yy, Var, levels=8, colors="white", linewidths=0.3, alpha=0.4)

    # Data points
    ax.scatter(*Z_train.T,  s=28, c="#1565C0", alpha=0.75, marker="o",
               label="Known defectives (train)", zorder=3, linewidths=0.4, edgecolors="white")
    ax.scatter(*Z_norm.T,   s=20, c="#2E7D32", alpha=0.6,  marker="^",
               label="Normals (val)",            zorder=3, linewidths=0.4, edgecolors="white")
    ax.scatter(*Z_novel.T,  s=55, c="#B71C1C", alpha=0.9,  marker="*",
               label=f"Novel: {fold['held_out_type']}",  zorder=4,
               linewidths=0.5, edgecolors="#FF8F00")

    ax.set_xlabel("PCA component 1 (visualization only)")
    ax.set_ylabel("PCA component 2 (visualization only)")
    ax.set_title(
        f"GP Posterior Variance — {fold['category']} / held-out: {fold['held_out_type']}\n"
        f"(2-D visualization GP; production model uses d={pca_dim})",
        fontsize=9.5,
    )
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.85)

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("GP posterior variance\n(↑ = more novel)", fontsize=9)

    plt.savefig(save_path)
    plt.close()
    log.info(f"Saved {save_path.name}")
    return save_path


# ---------------------------------------------------------------------------
# R — Kernel selection frequency
# ---------------------------------------------------------------------------

def plot_kernel_selection(
    fold_results: list[dict[str, Any]],
    save_path: Optional[Path] = None,
) -> Path:
    """Bar chart: how often each of the 6 kernels was selected as best by LML."""
    from collections import Counter

    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "R_kernel_selection.png"

    kernel_counts = Counter(r.get("kernel_name", "rbf") for r in fold_results)
    names  = list(kernel_counts.keys())
    counts = list(kernel_counts.values())
    total  = sum(counts)

    LABEL_MAP = {
        "rbf":     "RBF (SE)",
        "mat12":   "Matérn-½",
        "mat32":   "Matérn-³⁄₂",
        "mat52":   "Matérn-⁵⁄₂",
        "rq":      "Rational Q.",
        "lin_rbf": "Linear+RBF",
    }
    COLORS = ["#1565C0", "#2E7D32", "#F57F17", "#AD1457", "#6A1B9A", "#00695C"]
    color_map = dict(zip(["rbf","mat12","mat32","mat52","rq","lin_rbf"], COLORS))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(
        [LABEL_MAP.get(n, n) for n in names],
        [c / total * 100 for c in counts],
        color=[color_map.get(n, "#607D8B") for n in names],
        edgecolor="white", linewidth=0.8,
    )
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f"{c}", ha="center", fontsize=9)

    ax.set_ylabel("% of folds where kernel selected (LML)")
    ax.set_title("Kernel Selection Frequency across 72 LODTO Folds")
    ax.set_ylim(0, max(c / total * 100 for c in counts) + 10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    log.info(f"Saved {save_path.name}")
    return save_path


# ---------------------------------------------------------------------------
# R — PCA dimension sweep
# ---------------------------------------------------------------------------

def plot_pca_dim_sweep(
    results_by_dim: dict[int, list[dict[str, Any]]],
    save_path: Optional[Path] = None,
) -> Path:
    """
    AUROC (defect-only) vs PCA dimension d ∈ {8,12,16}.
    results_by_dim: {8: [...], 12: [...], 16: [...]} each list of scored fold dicts.
    """
    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "R_pca_dim_sweep.png"

    dims  = sorted(results_by_dim.keys())
    means = []
    stds  = []
    for d in dims:
        aucs = [r["auroc_do"] for r in results_by_dim[d]
                if not np.isnan(r.get("auroc_do", float("nan")))]
        means.append(np.mean(aucs) if aucs else 0)
        stds.append(np.std(aucs)  if aucs else 0)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(dims, means, yerr=stds, fmt="o-", color="#1565C0",
                capsize=5, linewidth=2, markersize=7, label="Mean AUROC ± SD")
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="Chance")
    ax.set_xticks(dims)
    ax.set_xlabel("PCA latent dimension d")
    ax.set_ylabel("AUROC (defect-only)")
    ax.set_title("PCA Dimensionality vs Novelty-Detection AUROC")
    ax.legend()
    ax.set_ylim(0.3, 1.0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    log.info(f"Saved {save_path.name}")
    return save_path


# ---------------------------------------------------------------------------
# Orchestration: run all method figures for phase 3
# ---------------------------------------------------------------------------

def run_method_figures(
    gp_results: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    device: Optional[str] = None,
    pca_dim: int = 16,
) -> None:
    """
    Called at the end of phase 3 to produce method-illustration figures.
    Picks the fold closest to median AUROC (if available) for M1.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── kernel selection frequency ────────────────────────────────────────
    if any("kernel_name" in r for r in gp_results):
        plot_kernel_selection(
            gp_results,
            save_path=FIGURES_DIR / f"R_kernel_selection_pca{pca_dim}.png",
        )

    # ── GP uncertainty surface ────────────────────────────────────────────
    # Pick fold with median n_train (avoids cherry-picking / worst-case)
    scored = [r for r in gp_results if r.get("n_train", 0) > 5]
    if not scored:
        log.warning("No scored folds with n_train > 5; skipping uncertainty surface.")
        return

    n_trains = [r["n_train"] for r in scored]
    median_idx = int(np.argsort(n_trains)[len(n_trains) // 2])
    rep_result = scored[median_idx]
    fold_map   = {f["fold_id"]: f for f in folds}
    rep_fold   = fold_map.get(rep_result["fold_id"])

    if rep_fold is not None:
        try:
            plot_gp_uncertainty_surface(
                rep_fold,
                save_path=FIGURES_DIR / f"M1_gp_uncertainty_surface_pca{pca_dim}.png",
                device=device,
                pca_dim=pca_dim,
            )
        except Exception as e:
            log.warning(f"Could not produce uncertainty surface: {e}")

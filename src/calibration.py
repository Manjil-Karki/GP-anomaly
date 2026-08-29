"""
GP calibration diagnostics — regression posterior and classification head.

Regression posterior:
  - RMSE on val_defective images (GP mean vs true log-severity)
  - Predictive interval coverage: fraction of val_def falling in 90% PI
  - PIT histogram: if calibrated, Φ((y - μ)/σ) is uniform on [0,1]

Classification head:
  - Brier score: mean squared error of predicted probability
  - ECE: expected calibration error (15-bin reliability diagram)
  - Reliability diagram saved as figure

These are the metrics promised in the proposal's Methodology section.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from .config import FIGURES_DIR, EVAL_DIR

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regression calibration
# ---------------------------------------------------------------------------

def regression_rmse(
    gp_mean_log: list[float],
    true_severity: list[float],
) -> float:
    """RMSE between GP mean (log-space) and true log-severity."""
    mu  = np.array(gp_mean_log, dtype=np.float64)
    y   = np.log(np.array(true_severity, dtype=np.float64))
    return float(np.sqrt(np.mean((mu - y) ** 2)))


def pit_values(
    gp_mean_log: np.ndarray,
    gp_var_log: np.ndarray,
    true_severity: np.ndarray,
) -> np.ndarray:
    """
    Probability Integral Transform values.
    PIT_i = Φ((y_i - μ_i) / σ_i) where Φ is the standard normal CDF.
    Uniform PIT histogram → well-calibrated predictive distribution.
    """
    from scipy.stats import norm
    sigma = np.sqrt(np.clip(gp_var_log, 1e-12, None))
    y_log = np.log(np.clip(true_severity, 1e-12, None))
    return norm.cdf((y_log - gp_mean_log) / sigma)


def interval_coverage(
    gp_mean_log: np.ndarray,
    gp_var_log: np.ndarray,
    true_severity: np.ndarray,
    alpha: float = 0.90,
) -> float:
    """
    Fraction of val_defective images whose true log-severity falls within
    the (alpha)% predictive interval of the GP. Target: ≈ alpha if calibrated.
    """
    from scipy.stats import norm
    z     = norm.ppf((1 + alpha) / 2)
    sigma = np.sqrt(np.clip(gp_var_log, 1e-12, None))
    y_log = np.log(np.clip(true_severity, 1e-12, None))
    lo    = gp_mean_log - z * sigma
    hi    = gp_mean_log + z * sigma
    return float(np.mean((y_log >= lo) & (y_log <= hi)))


# ---------------------------------------------------------------------------
# Classification head calibration
# ---------------------------------------------------------------------------

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between predicted probability and true binary label."""
    return float(np.mean((probs - labels) ** 2))


def ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error: weighted mean of |confidence - accuracy| per bin.
    probs: predicted probability of positive class per image.
    labels: ground-truth binary labels.
    """
    bins      = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx   = np.digitize(probs, bins, right=True).clip(1, n_bins) - 1
    ece_val   = 0.0
    n         = len(probs)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece_val += mask.sum() / n * abs(conf - acc)
    return float(ece_val)


# ---------------------------------------------------------------------------
# Per-fold calibration (aggregated from fold result dicts)
# ---------------------------------------------------------------------------

def calibrate_folds(
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute calibration metrics across all folds that contain GP mean predictions.
    Returns aggregated metrics and per-fold lists for plotting.
    """
    rmse_list     = []
    coverage_list = []
    all_pit       = []

    for r in fold_results:
        if "gp_mean_val_def" not in r or "val_severity" not in r:
            continue

        mu      = np.array(r["gp_mean_val_def"], dtype=np.float64)
        sev     = np.array(r["val_severity"],     dtype=np.float64)
        known_v = np.array(r["known_scores"],     dtype=np.float64)  # GP variance on val_def

        rmse_list.append(regression_rmse(mu.tolist(), sev.tolist()))

        cov = interval_coverage(mu, known_v, sev, alpha=0.90)
        coverage_list.append(cov)

        pit = pit_values(mu, known_v, sev)
        all_pit.extend(pit.tolist())

    return {
        "mean_rmse":              float(np.nanmean(rmse_list)) if rmse_list else float("nan"),
        "std_rmse":               float(np.nanstd(rmse_list))  if rmse_list else float("nan"),
        "mean_90pct_coverage":    float(np.nanmean(coverage_list)) if coverage_list else float("nan"),
        "pit_values":             all_pit,
        "n_folds_calibrated":     len(rmse_list),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_pit_histogram(
    pit_values: list[float],
    save_path: Path | None = None,
) -> Path:
    """
    PIT histogram. A well-calibrated GP produces a uniform distribution.
    Skew left → overconfident; skew right → underconfident.
    """
    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "C1_pit_histogram.png"

    pit = np.array(pit_values)
    n_bins = 20
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(pit, bins=n_bins, density=True, color="#1565C0", alpha=0.75, edgecolor="white")
    ax.axhline(1.0, ls="--", color="gray", lw=1.2, label="Uniform (calibrated)")
    ax.set_xlabel("PIT value  Φ((y − μ) / σ)")
    ax.set_ylabel("Density")
    ax.set_title(f"PIT Histogram — GP Regression Posterior\n"
                 f"(n = {len(pit)} val_defective images across all folds)")
    ax.legend()
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {save_path.name}")
    return save_path


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    save_path: Path | None = None,
    n_bins: int = 10,
) -> Path:
    """
    Reliability diagram for GP classification head.
    Diagonal → perfect calibration. Above diagonal → underconfident.
    """
    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "C2_reliability_diagram.png"

    bins    = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(probs, bins, right=True).clip(1, n_bins) - 1
    mean_conf, mean_acc, counts = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        mean_conf.append(probs[mask].mean())
        mean_acc.append(labels[mask].mean())
        counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    ax.bar(mean_conf, mean_acc, width=0.07, alpha=0.6, color="#1565C0",
           label="GP classification head")
    ax.scatter(mean_conf, mean_acc, s=40, color="#1565C0", zorder=3)
    ece_val = ece(probs, labels, n_bins=15)
    ax.set_xlabel("Mean confidence (predicted probability)")
    ax.set_ylabel("Fraction positive (actual accuracy)")
    ax.set_title(f"Reliability Diagram\nECE = {ece_val:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {save_path.name}")
    return save_path


def _pool_classification_data(
    fold_results: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pool GP variance scores from the validation split across all folds and
    convert to proxy probabilities for the reliability diagram (C2) and ECE.

    GP variance is not a probability.  We use per-fold min-max normalisation
    on the val set to map variance → [0, 1], then pool across folds.
    Labels: 1 = val_defective (known defect, novelty-positive proxy),
            0 = val_normal.
    """
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for r in fold_results:
        if "val_scores" not in r or "val_labels" not in r:
            continue
        sc = np.array(r["val_scores"], dtype=np.float64)
        lb = np.array(r["val_labels"], dtype=np.float64)
        lo, hi = sc.min(), sc.max()
        probs = np.clip((sc - lo) / (hi - lo + 1e-12), 0.0, 1.0)
        all_probs.append(probs)
        all_labels.append(lb)
    if not all_probs:
        return np.array([]), np.array([])
    return np.concatenate(all_probs), np.concatenate(all_labels)


def run_calibration(
    fold_results: list[dict[str, Any]],
    save_dir: Path = EVAL_DIR,
) -> dict[str, Any]:
    """
    Entry point called from pipeline phase 4.
    Computes all calibration metrics and saves figures + JSON.
    Produces:
      C1 — PIT histogram (GP regression posterior)
      C2 — Reliability diagram + ECE + Brier score (classification head)
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cal = calibrate_folds(fold_results)

    if cal["pit_values"]:
        plot_pit_histogram(cal["pit_values"])

    # C2: classification calibration using normalised GP variance as prob proxy
    probs, labels = _pool_classification_data(fold_results)
    bs_val  = float("nan")
    ece_val = float("nan")
    if len(probs) > 0:
        bs_val  = brier_score(probs, labels)
        ece_val = ece(probs, labels, n_bins=15)
        plot_reliability_diagram(probs, labels)
        log.info(f"Classification calibration: ECE={ece_val:.4f}  Brier={bs_val:.4f}")

    summary = {k: v for k, v in cal.items() if k != "pit_values"}
    summary["target_90pct_coverage"] = 0.90
    summary["brier_score"]           = bs_val
    summary["ece"]                   = ece_val

    out = save_dir / "calibration_summary.json"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Calibration: RMSE={cal['mean_rmse']:.4f}, "
             f"90% coverage={cal['mean_90pct_coverage']:.3f} (target 0.90)")
    return summary

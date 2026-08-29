"""Metrics, statistical tests, and results aggregation."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import EVAL_DIR, SIGNIFICANCE_LEVEL
from .threshold import optimise_threshold, apply_threshold

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


# ---------------------------------------------------------------------------
# AUROC under two evaluation labellings
# ---------------------------------------------------------------------------

def defect_only_auroc(
    novel_scores: np.ndarray,
    known_scores: np.ndarray,
) -> float:
    """
    Primary metric: novel defect type (label=1) vs known defect types (label=0).
    Tests whether GP variance discriminates novelty independently of normal vs defect.
    """
    scores = np.concatenate([novel_scores, known_scores])
    labels = np.concatenate([np.ones(len(novel_scores)), np.zeros(len(known_scores))])
    return auroc(scores, labels)


def inclusive_auroc(
    novel_scores: np.ndarray,
    normal_scores: np.ndarray,
) -> float:
    """
    Secondary metric: novel defect (label=1) vs normal images (label=0).
    Reflects operational novelty detection performance.
    """
    scores = np.concatenate([novel_scores, normal_scores])
    labels = np.concatenate([np.ones(len(novel_scores)), np.zeros(len(normal_scores))])
    return auroc(scores, labels)


# ---------------------------------------------------------------------------
# Per-fold scoring
# ---------------------------------------------------------------------------

def score_fold(
    gp_result: dict[str, Any],
    known_scores: np.ndarray,
    normal_scores: np.ndarray,
) -> dict[str, Any]:
    """
    Compute all GP metrics for one fold including AUPR, RMSE, and F1.
    known_scores: GP variances for known defect types (same category, test split).
    normal_scores: GP variances for normal images (val_normal split is fine here).
    """
    test_scores = np.array(gp_result["test_scores"])
    val_scores  = np.array(gp_result["val_scores"])
    val_labels  = np.array(gp_result["val_labels"])

    threshold = optimise_threshold(val_scores, val_labels)
    preds     = apply_threshold(test_scores, threshold)

    # AUPR (defect-only labelling)
    do_scores = np.concatenate([test_scores, known_scores])
    do_labels = np.concatenate([np.ones(len(test_scores)), np.zeros(len(known_scores))])
    aupr_val  = auprc(do_scores, do_labels)

    # F1 on test split (all test images are novel = positive class)
    tp = int(preds.sum())
    fn = int((1 - preds).sum())
    # FP: known-type images above threshold (if known_scores available)
    fp = int((known_scores >= threshold).sum()) if len(known_scores) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # RMSE of GP mean on val_defective (regression calibration)
    rmse = float("nan")
    if "gp_mean_val_def" in gp_result and "val_severity" in gp_result:
        mu    = np.array(gp_result["gp_mean_val_def"], dtype=np.float64)
        y_log = np.log(np.clip(np.array(gp_result["val_severity"], dtype=np.float64), 1e-12, None))
        rmse  = float(np.sqrt(np.mean((mu - y_log) ** 2)))

    return {
        "fold_id":           gp_result["fold_id"],
        "category":          gp_result["category"],
        "held_out_type":     gp_result["held_out_type"],
        "pca_dim":           gp_result["pca_dim"],
        "threshold":         float(threshold),
        "n_test":            len(test_scores),
        "detection_rate":    float(preds.mean()),
        "auroc_do":          defect_only_auroc(test_scores, known_scores),
        "auroc_incl":        inclusive_auroc(test_scores, normal_scores),
        "aupr_do":           aupr_val,
        "f1":                float(f1),
        "precision":         float(precision),
        "recall":            float(recall),
        "rmse_log_severity": rmse,
        "lml":               gp_result["lml"],
        "pca_var_explained": gp_result.get("pca_var_explained", float("nan")),
    }


# ---------------------------------------------------------------------------
# Multi-method per-fold scoring
# ---------------------------------------------------------------------------

_METHOD_KEYS: dict[str, tuple[str, str, str]] = {
    "gp":    ("val_scores",        "known_scores",       "test_scores"),
    "knn":   ("knn_val_scores",    "knn_known_scores",   "knn_test_scores"),
    "maha":  ("maha_val_scores",   "maha_known_scores",  "maha_test_scores"),
    "fused": ("fused_val_scores",  "fused_known_scores", "fused_test_scores"),
}


def score_all_methods(
    gp_result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Score all available novelty methods (GP, kNN, Mahal, fused) for one fold.
    Returns a dict keyed by method name; absent keys mean the fold predates
    those scores (run phase 3 again with resume=True to fill them).
    """
    val_labels = np.array(gp_result["val_labels"])
    out: dict[str, dict[str, Any]] = {}

    for method, (val_key, known_key, test_key) in _METHOD_KEYS.items():
        if val_key not in gp_result:
            continue
        val_sc   = np.array(gp_result[val_key])
        known_sc = np.array(gp_result[known_key])
        test_sc  = np.array(gp_result[test_key])
        norm_sc  = val_sc[val_labels == 0]

        threshold = optimise_threshold(val_sc, val_labels)
        preds     = apply_threshold(test_sc, threshold)

        do_scores = np.concatenate([test_sc, known_sc])
        do_labels = np.concatenate([np.ones(len(test_sc)), np.zeros(len(known_sc))])

        out[method] = {
            "method":         method,
            "fold_id":        gp_result["fold_id"],
            "category":       gp_result["category"],
            "held_out_type":  gp_result["held_out_type"],
            "pca_dim":        gp_result["pca_dim"],
            "threshold":      float(threshold),
            "n_test":         len(test_sc),
            "detection_rate": float(preds.mean()),
            "auroc_do":       defect_only_auroc(test_sc, known_sc),
            "auroc_incl":     inclusive_auroc(test_sc, norm_sc),
            "aupr_do":        auprc(do_scores, do_labels),
        }
    return out


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def wilcoxon_test(
    gp_aucs: np.ndarray,
    baseline_aucs: np.ndarray,
    name: str,
    alpha: float = SIGNIFICANCE_LEVEL,
) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test on matched fold AUROCs."""
    diffs = gp_aucs - baseline_aucs
    if np.all(diffs == 0) or np.isnan(diffs).all():
        return {"method": name, "statistic": 0.0, "p_value": 1.0, "significant": False}
    valid = ~np.isnan(diffs)
    stat, p = wilcoxon(gp_aucs[valid], baseline_aucs[valid], alternative="two-sided")
    return {"method": name, "statistic": float(stat), "p_value": float(p), "significant": bool(p < alpha)}


def holm_correct(tests: list[dict[str, Any]], alpha: float = SIGNIFICANCE_LEVEL) -> list[dict]:
    """Holm-Bonferroni correction across multiple comparisons."""
    ranked = sorted(tests, key=lambda d: d["p_value"])
    m = len(ranked)
    for k, t in enumerate(ranked):
        corrected = alpha / (m - k)
        t["p_holm"] = float(min(t["p_value"] * (m - k), 1.0))
        t["significant_holm"] = bool(t["p_value"] < corrected)
    return ranked


# ---------------------------------------------------------------------------
# Results tables
# ---------------------------------------------------------------------------

def build_fold_table(scored_folds: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(scored_folds)


def summarise(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Per-category mean ± std of AUROC (defect-only)."""
    grp = fold_df.groupby("category")["auroc_do"]
    return pd.DataFrame({
        "mean_auroc_do": grp.mean(),
        "std_auroc_do":  grp.std(),
        "n_folds":       grp.count(),
    }).reset_index()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_json(obj: Any, name: str, directory: Path = EVAL_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"{name}.json"
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)
    return p


def load_json(name: str, directory: Path = EVAL_DIR) -> Any:
    with open(directory / f"{name}.json") as f:
        return json.load(f)

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
    Compute all metrics for one fold.
    known_scores: GP variances for known defect types (same category, test split).
    normal_scores: GP variances for normal images (val_normal split is fine here).
    """
    test_scores = np.array(gp_result["test_scores"])
    val_scores  = np.array(gp_result["val_scores"])
    val_labels  = np.array(gp_result["val_labels"])

    threshold = optimise_threshold(val_scores, val_labels)
    preds     = apply_threshold(test_scores, threshold)

    return {
        "fold_id":          gp_result["fold_id"],
        "category":         gp_result["category"],
        "held_out_type":    gp_result["held_out_type"],
        "pca_dim":          gp_result["pca_dim"],
        "threshold":        threshold,
        "n_test":           len(test_scores),
        "detection_rate":   float(preds.mean()),
        "auroc_do":         defect_only_auroc(test_scores, known_scores),
        "auroc_incl":       inclusive_auroc(test_scores, normal_scores),
        "lml":              gp_result["lml"],
        "pca_var_explained": gp_result.get("pca_var_explained", float("nan")),
    }


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
    return {"method": name, "statistic": float(stat), "p_value": float(p), "significant": p < alpha}


def holm_correct(tests: list[dict[str, Any]], alpha: float = SIGNIFICANCE_LEVEL) -> list[dict]:
    """Holm-Bonferroni correction across multiple comparisons."""
    ranked = sorted(tests, key=lambda d: d["p_value"])
    m = len(ranked)
    for k, t in enumerate(ranked):
        corrected = alpha / (m - k)
        t["p_holm"] = min(t["p_value"] * (m - k), 1.0)
        t["significant_holm"] = t["p_value"] < corrected
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

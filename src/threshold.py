"""Cost-optimal threshold selection on the validation fold."""
from __future__ import annotations

import numpy as np

from .config import C_FN, C_FP


def cost_at_threshold(
    t: float,
    scores: np.ndarray,
    labels: np.ndarray,
    c_fn: float = C_FN,
    c_fp: float = C_FP,
) -> float:
    """C(t) = c_FN · P(missed novel) + c_FP · P(false alarm)."""
    preds   = (scores >= t).astype(int)
    n_pos   = labels.sum()
    n_neg   = (1 - labels).sum()

    p_missed = ((labels == 1) & (preds == 0)).sum() / n_pos if n_pos > 0 else 0.0
    p_alarm  = ((labels == 0) & (preds == 1)).sum() / n_neg if n_neg > 0 else 0.0

    return c_fn * p_missed + c_fp * p_alarm


def optimise_threshold(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    c_fn: float = C_FN,
    c_fp: float = C_FP,
    n_candidates: int = 500,
) -> float:
    """
    Grid search over n_candidates thresholds ∈ [min, max] of val_scores.
    Returns the threshold that minimises the asymmetric cost function.
    """
    candidates = np.linspace(val_scores.min(), val_scores.max(), n_candidates)
    costs = np.array([
        cost_at_threshold(t, val_scores, val_labels, c_fn, c_fp)
        for t in candidates
    ])
    return float(candidates[np.argmin(costs)])


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Binary predictions: 1 = novel defect detected, 0 = not detected."""
    return (scores >= threshold).astype(int)


def cost_sweep(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    ratios: list[float] | None = None,
) -> list[dict[str, float]]:
    """
    Sweep C_FN/C_FP ratios (C_FP fixed at 1.0) and return the cost-optimal
    operating point for each ratio.  Used to produce the cost-operating curve
    promised in the proposal.
    """
    if ratios is None:
        ratios = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    rows = []
    for ratio in ratios:
        c_fn  = float(ratio)
        t     = optimise_threshold(val_scores, val_labels, c_fn=c_fn, c_fp=1.0)
        preds = apply_threshold(val_scores, t)
        n_pos = int(val_labels.sum())
        n_neg = int((1 - val_labels).sum())
        p_miss  = float(((val_labels == 1) & (preds == 0)).sum() / n_pos) if n_pos else 0.0
        p_alarm = float(((val_labels == 0) & (preds == 1)).sum() / n_neg) if n_neg else 0.0
        rows.append({
            "c_fn_c_fp_ratio":  float(ratio),
            "threshold":        float(t),
            "miss_rate":        p_miss,
            "false_alarm_rate": p_alarm,
            "cost":             c_fn * p_miss + 1.0 * p_alarm,
        })
    return rows

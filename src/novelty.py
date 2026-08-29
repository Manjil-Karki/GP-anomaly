"""Novelty scoring: GP posterior variance + k-NN + Mahalanobis + rank fusion."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import GP_RESULTS_DIR, EMBEDDINGS_DIR
from .embeddings import load_embeddings
from .folds import load_folds
from .gp_model import (
    FittedGP, fit_gp, predict_gp,
    gp_class_probability,
    FittedGPClassifier, fit_gp_type_classifier, predict_gp_type_entropy,
)
from .novelty_scores import knn_score, mahalanobis_score, rank_fuse
from .pca_transform import FoldPCA, fit_fold_pca

log = logging.getLogger(__name__)

# k for k-NN distance scorer
KNN_K = 5


def _score_split(
    X: np.ndarray,
    X_train: np.ndarray,
    fitted: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (gp_var, knn_dist, maha_dist) for one split."""
    _, gp_var = predict_gp(X.astype(np.float64), fitted)
    knn       = knn_score(X, X_train, k=KNN_K)
    maha      = mahalanobis_score(X, X_train)
    return gp_var, knn, maha


def run_single_fold(
    fold: dict[str, Any],
    pca_dim: int,
    device: Optional[str] = None,
    save_pca: bool = True,
    pca_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Full novelty pipeline for one LODTO fold.

    Steps:
      1. Load cached DINOv2 embeddings for all splits.
      2. Fit per-fold PCA on train_defective only (no leakage).
      3. Fit ExactGP on (X_train_pca, log(severity)).
      4. Score val_defective, val_normal, test_defective with:
           - GP posterior variance  (original method)
           - k-NN distance          (Sun et al. ICML 2022)
           - Mahalanobis distance   (Lee et al. NeurIPS 2018)
           - Rank-fused combination (proposed method)
      5. Capture GP mean predictions on val_defective for RMSE.
    """
    train_paths    = fold["train_defective"]
    val_def_paths  = fold["val_defective"]
    val_norm_paths = fold["val_normal"]
    test_paths     = fold["test_defective"]

    # ── 1. Embeddings ─────────────────────────────────────────────────────
    X_train    = load_embeddings(train_paths)
    X_val_def  = load_embeddings(val_def_paths)
    X_val_norm = load_embeddings(val_norm_paths)
    X_test     = load_embeddings(test_paths)

    # ── 2. PCA (fitted on training defectives only) ────────────────────────
    fpca = fit_fold_pca(X_train, pca_dim)
    X_tr = fpca.transform(X_train)
    X_vd = fpca.transform(X_val_def)
    X_vn = fpca.transform(X_val_norm)
    X_te = fpca.transform(X_test)

    if save_pca and pca_dir is not None:
        pca_path = pca_dir / f"{fold['fold_id']}_pca.pkl"
        fpca.save(pca_path)

    # ── 3. GP on log(severity) ─────────────────────────────────────────────
    y_train = np.log(np.array(fold["train_severity"], dtype=np.float64))
    fitted  = fit_gp(X_tr.astype(np.float64), y_train, device=device)

    # GP mean on val_defective — used for RMSE calibration check
    gp_mean_vd, _ = predict_gp(X_vd.astype(np.float64), fitted)

    # ── 4. Score all splits with all methods ──────────────────────────────
    gp_var_vd,  knn_vd,  maha_vd  = _score_split(X_vd, X_tr, fitted)
    gp_var_vn,  knn_vn,  maha_vn  = _score_split(X_vn, X_tr, fitted)
    gp_var_te,  knn_te,  maha_te  = _score_split(X_te, X_tr, fitted)

    # Rank-fused scores: ranks computed JOINTLY across all three splits so
    # the cross-split ordering (novel > known) is preserved for AUROC.
    # Per-split rank normalisation destroys this ordering (both val_def and
    # test end up uniform on [0,1] independently → AUROC ≈ 0.5).
    n_vd, n_vn = len(gp_var_vd), len(gp_var_vn)
    _fused_all = rank_fuse(
        np.concatenate([gp_var_vd, gp_var_vn, gp_var_te]),
        np.concatenate([knn_vd,    knn_vn,    knn_te]),
        np.concatenate([maha_vd,   maha_vn,   maha_te]),
    )
    fused_vd = _fused_all[:n_vd]
    fused_vn = _fused_all[n_vd:n_vd + n_vn]
    fused_te = _fused_all[n_vd + n_vn:]

    # ── 4a. Enhancement A — binary GP class probability for C2 ────────────
    # Threshold at median log-severity of training set (≈ 50% positive class).
    log_thresh = float(np.median(y_train))
    gp_cls_prob_vd = gp_class_probability(X_vd.astype(np.float64), fitted, log_thresh)
    val_def_sev    = np.array(fold["val_severity"], dtype=np.float64)
    gp_cls_labels_vd = (np.log(np.clip(val_def_sev, 1e-12, None)) > log_thresh).astype(np.float64)

    # ── 4b. Enhancement B — multi-class GP type classifier ────────────────
    gpclf_entropy_vd = gpclf_entropy_vn = gpclf_entropy_te = None
    try:
        from pathlib import Path as _Path
        type_names  = [_Path(p).parent.name for p in train_paths]
        unique_types = sorted(set(type_names))
        if len(unique_types) >= 2:
            t2i         = {t: i for i, t in enumerate(unique_types)}
            type_labels = np.array([t2i[t] for t in type_names], dtype=np.int64)
            fitted_clf  = fit_gp_type_classifier(
                X_tr.astype(np.float64), type_labels, device=device
            )
            gpclf_entropy_vd = predict_gp_type_entropy(X_vd.astype(np.float64), fitted_clf)
            gpclf_entropy_vn = predict_gp_type_entropy(X_vn.astype(np.float64), fitted_clf)
            gpclf_entropy_te = predict_gp_type_entropy(X_te.astype(np.float64), fitted_clf)
            log.info(f"  GP classifier: K={len(unique_types)} types, "
                     f"test_entropy={gpclf_entropy_te.mean():.3f} "
                     f"val_def_entropy={gpclf_entropy_vd.mean():.3f}")
        else:
            log.warning("  GP classifier skipped: only 1 unique type in training fold")
    except Exception as exc:
        log.warning(f"  GP classifier failed: {exc}")

    # 4-way rank fusion (GP var + kNN + Maha + GP classifier entropy)
    if gpclf_entropy_te is not None:
        _fused5_all = rank_fuse(
            np.concatenate([gp_var_vd, gp_var_vn, gp_var_te]),
            np.concatenate([knn_vd,    knn_vn,    knn_te]),
            np.concatenate([maha_vd,   maha_vn,   maha_te]),
            np.concatenate([gpclf_entropy_vd, gpclf_entropy_vn, gpclf_entropy_te]),
        )
        fused5_vd = _fused5_all[:n_vd]
        fused5_vn = _fused5_all[n_vd:n_vd + n_vn]
        fused5_te = _fused5_all[n_vd + n_vn:]
    else:
        fused5_vd = fused5_vn = fused5_te = None

    # ── 5. Combine val splits for threshold optimisation ──────────────────
    # Labels: 1 = known defective (positive for novelty task), 0 = normal
    val_labels = np.concatenate([
        np.ones(len(gp_var_vd)),
        np.zeros(len(gp_var_vn)),
    ])

    def _val(a, b):
        return np.concatenate([a, b]).tolist()

    result: dict = {
        # ── Fold metadata ─────────────────────────────────────────────────
        "fold_id":           fold["fold_id"],
        "category":          fold["category"],
        "held_out_type":     fold["held_out_type"],
        "pca_dim":           pca_dim,
        "kernel_name":       fitted.kernel_name,
        "lml":               float(fitted.lml),
        "pca_var_explained": float(fpca.explained_variance_ratio.sum()),
        "n_train":           len(train_paths),
        "n_test":            len(test_paths),

        # ── GP method ────────────────────────────────────────────────────
        "val_scores":        _val(gp_var_vd, gp_var_vn),
        "val_labels":        val_labels.tolist(),
        "known_scores":      gp_var_vd.tolist(),
        "test_scores":       gp_var_te.tolist(),

        # ── k-NN method (Sun et al. 2022) ────────────────────────────────
        "knn_val_scores":    _val(knn_vd, knn_vn),
        "knn_known_scores":  knn_vd.tolist(),
        "knn_test_scores":   knn_te.tolist(),

        # ── Mahalanobis method (Lee et al. 2018) ─────────────────────────
        "maha_val_scores":   _val(maha_vd, maha_vn),
        "maha_known_scores": maha_vd.tolist(),
        "maha_test_scores":  maha_te.tolist(),

        # ── Rank-fused method (proposed) ──────────────────────────────────
        "fused_val_scores":   _val(fused_vd, fused_vn),
        "fused_known_scores": fused_vd.tolist(),
        "fused_test_scores":  fused_te.tolist(),

        # ── GP regression outputs (for RMSE + calibration) ────────────────
        "gp_mean_val_def":   gp_mean_vd.tolist(),
        "val_severity":      fold["val_severity"],
        "test_severity":     fold["test_severity"],
        "test_defect_types": fold["test_defect_types"],

        # ── Enhancement A: real GP binary class probabilities (for C2) ────
        "gp_class_prob_val_def":    gp_cls_prob_vd.tolist(),
        "gp_class_labels_val_def":  gp_cls_labels_vd.tolist(),
        "severity_log_threshold":   log_thresh,
    }

    # Enhancement B: GP type classifier entropy scores (optional)
    if gpclf_entropy_te is not None:
        result.update({
            "gpclf_val_scores":    _val(gpclf_entropy_vd, gpclf_entropy_vn),
            "gpclf_known_scores":  gpclf_entropy_vd.tolist(),
            "gpclf_test_scores":   gpclf_entropy_te.tolist(),
            "fused5_val_scores":   _val(fused5_vd, fused5_vn),
            "fused5_known_scores": fused5_vd.tolist(),
            "fused5_test_scores":  fused5_te.tolist(),
        })

    return result


def run_all_folds(
    pca_dim: int = 16,
    device: Optional[str] = None,
    save_dir: Path = GP_RESULTS_DIR,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """
    Run all LODTO folds for a given PCA dimension.
    resume=True skips folds that already have all four novelty score keys.
    """
    pca_dir = save_dir / f"pca{pca_dim}" / "pca_objects"
    out_dir = save_dir / f"pca{pca_dim}"
    pca_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds   = load_folds()
    results = []

    for i, fold in enumerate(folds):
        fid      = fold["fold_id"]
        out_path = out_dir / f"{fid}.json"

        if resume and out_path.exists():
            with open(out_path) as f:
                cached = json.load(f)
            # Skip only if both fused and GP-classifier keys are present
            if "fused_test_scores" in cached and "gpclf_test_scores" in cached:
                log.info(f"[{i+1}/{len(folds)}] Skip {fid} (cached)")
                results.append(cached)
                continue

        log.info(f"[{i+1}/{len(folds)}] {fid}  pca_dim={pca_dim}")
        result = run_single_fold(fold, pca_dim, device=device,
                                 save_pca=True, pca_dir=pca_dir)

        with open(out_path, "w") as f:
            json.dump(result, f)
        results.append(result)

    log.info(f"GP+kNN+Mahal complete: {len(results)} folds, pca_dim={pca_dim}.")
    return results

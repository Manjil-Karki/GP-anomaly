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
from .novelty_scores import (
    knn_score, mahalanobis_score, prototype_distance_score,
    lof_score, rank_fuse, zscore_fuse,
)
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


def _fit_lof(X_train: np.ndarray, k: int = 5):
    """Fit Isolation Forest on training set. Returns fitted clf."""
    from sklearn.ensemble import IsolationForest
    n_est = min(200, max(50, len(X_train) * 2))
    clf = IsolationForest(n_estimators=n_est, random_state=42, contamination="auto")
    clf.fit(X_train.astype(np.float64))
    return clf


def _lof_predict(clf, X: np.ndarray) -> np.ndarray:
    if clf is None:
        return np.zeros(len(X), dtype=np.float64)
    return (-clf.score_samples(X.astype(np.float64))).astype(np.float64)


def _calibrate_variance_scale(
    gp_mean_vd: np.ndarray,
    gp_var_vd: np.ndarray,
    val_def_sev: np.ndarray,
    target: float = 0.90,
) -> float:
    """
    Find T ≥ 1 such that (target)% of val_def images fall inside the
    T-scaled (target)% GP predictive interval.  Fixes ExactGP overconfidence.
    Returns T=1.0 if the data are already calibrated or if search fails.
    """
    from scipy.optimize import brentq
    from scipy.stats import norm as _norm
    z = _norm.ppf((1 + target) / 2)
    y_log = np.log(np.clip(val_def_sev, 1e-12, None))

    def coverage(T: float) -> float:
        sigma = T * np.sqrt(np.clip(gp_var_vd, 1e-12, None))
        lo = gp_mean_vd - z * sigma
        hi = gp_mean_vd + z * sigma
        return float(np.mean((y_log >= lo) & (y_log <= hi))) - target

    try:
        # If already over-covered at T=1.0, no inflation needed
        if coverage(1.0) >= 0:
            return 1.0
        T_opt = brentq(coverage, 1.0, 200.0, xtol=1e-3)
        return float(T_opt)
    except (ValueError, RuntimeError):
        return 1.0


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

    # LOF: fit once on training set, apply to all splits
    _lof_clf = _fit_lof(X_tr, k=KNN_K)
    lof_vd   = _lof_predict(_lof_clf, X_vd)
    lof_vn   = _lof_predict(_lof_clf, X_vn)
    lof_te   = _lof_predict(_lof_clf, X_te)

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
    log_thresh  = float(np.median(y_train))
    val_def_sev = np.array(fold["val_severity"], dtype=np.float64)
    gp_cls_labels_vd = (np.log(np.clip(val_def_sev, 1e-12, None)) > log_thresh).astype(np.float64)

    # Post-hoc variance calibration: find T ≥ 1 that inflates GP sigma until
    # 90% of val_def images fall within the 90% predictive interval.
    # Fixes systematic ExactGP overconfidence without retraining.
    T_cal = _calibrate_variance_scale(gp_mean_vd, gp_var_vd, val_def_sev, target=0.90)
    gp_cls_prob_vd = gp_class_probability(
        X_vd.astype(np.float64), fitted, log_thresh, variance_scale=T_cal
    )
    log.info(f"  Variance calibration: T={T_cal:.3f}")

    # ── 4b. Enhancement B — multi-class GP type classifier ────────────────
    # Use a lower PCA sub-dimension for the classifier: with 20-120 training
    # images split across K types, 32-dim ARD is severely underdetermined.
    # Restricting to the top 16 components gives cleaner class boundaries.
    CLF_PCA_DIM = min(pca_dim, 16)
    gpclf_entropy_vd = gpclf_entropy_vn = gpclf_entropy_te = None
    proto_vd = proto_vn = proto_te = None
    type_labels_arr = None
    try:
        from pathlib import Path as _Path
        type_names   = [_Path(p).parent.name for p in train_paths]
        unique_types = sorted(set(type_names))
        if len(unique_types) >= 2:
            t2i             = {t: i for i, t in enumerate(unique_types)}
            type_labels_arr = np.array([t2i[t] for t in type_names], dtype=np.int64)

            # Prototype distance (per-class Maha) — fast, no GP needed
            proto_vd = prototype_distance_score(X_vd[:, :CLF_PCA_DIM], X_tr[:, :CLF_PCA_DIM], type_labels_arr)
            proto_vn = prototype_distance_score(X_vn[:, :CLF_PCA_DIM], X_tr[:, :CLF_PCA_DIM], type_labels_arr)
            proto_te = prototype_distance_score(X_te[:, :CLF_PCA_DIM], X_tr[:, :CLF_PCA_DIM], type_labels_arr)

            # DirichletGPC: 200 iterations, 3 restarts — keep run with lowest
            # mean training entropy (= most confident on known types)
            best_clf        = None
            best_train_ent  = np.inf
            for _restart in range(3):
                try:
                    _clf = fit_gp_type_classifier(
                        X_tr[:, :CLF_PCA_DIM].astype(np.float64),
                        type_labels_arr,
                        device=device,
                        max_iter=200,
                    )
                    _tr_ent = predict_gp_type_entropy(
                        X_tr[:, :CLF_PCA_DIM].astype(np.float64), _clf
                    ).mean()
                    if _tr_ent < best_train_ent:
                        best_train_ent = _tr_ent
                        best_clf = _clf
                except Exception:
                    pass

            if best_clf is not None:
                gpclf_entropy_vd = predict_gp_type_entropy(X_vd[:, :CLF_PCA_DIM].astype(np.float64), best_clf)
                gpclf_entropy_vn = predict_gp_type_entropy(X_vn[:, :CLF_PCA_DIM].astype(np.float64), best_clf)
                gpclf_entropy_te = predict_gp_type_entropy(X_te[:, :CLF_PCA_DIM].astype(np.float64), best_clf)
                log.info(f"  GP classifier: K={len(unique_types)} types, "
                         f"clf_pca={CLF_PCA_DIM}, "
                         f"test_entropy={gpclf_entropy_te.mean():.3f} "
                         f"known_entropy={gpclf_entropy_vd.mean():.3f} "
                         f"gap={gpclf_entropy_te.mean()-gpclf_entropy_vd.mean():+.3f}")
        else:
            log.warning("  GP classifier skipped: only 1 unique type in training fold")
    except Exception as exc:
        log.warning(f"  GP classifier failed: {exc}")

    # ── 4c. Embedding distance (proposal §7: shift analysis) ──────────────
    # Mean distance from test embeddings to nearest training embedding.
    # Proposal requires: "calibration tracked against shift, measured as the
    # mean embedding distance between the held-out type and the training types."
    try:
        from sklearn.neighbors import NearestNeighbors as _NN
        nn_emb = _NN(n_neighbors=1, metric="euclidean").fit(X_tr.astype(np.float64))
        emb_dist = float(nn_emb.kneighbors(X_te.astype(np.float64))[0].mean())
    except Exception:
        emb_dist = float("nan")

    # fused5 (4-way: GP var + kNN + Maha + GPclf entropy)
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

    # fused3 (best-pair + prototype: GPclf + Maha + Proto)
    if gpclf_entropy_te is not None and proto_te is not None:
        _fused3_all = rank_fuse(
            np.concatenate([gpclf_entropy_vd, gpclf_entropy_vn, gpclf_entropy_te]),
            np.concatenate([maha_vd,          maha_vn,          maha_te]),
            np.concatenate([proto_vd,         proto_vn,         proto_te]),
        )
        fused3_vd = _fused3_all[:n_vd]
        fused3_vn = _fused3_all[n_vd:n_vd + n_vn]
        fused3_te = _fused3_all[n_vd + n_vn:]
    else:
        fused3_vd = fused3_vn = fused3_te = None

    # fused2z: zscore_fuse(GPclf + Maha) — preserves magnitude over rank_fuse
    if gpclf_entropy_te is not None:
        _fused2z_all = zscore_fuse(
            np.concatenate([gpclf_entropy_vd, gpclf_entropy_vn, gpclf_entropy_te]),
            np.concatenate([maha_vd,          maha_vn,          maha_te]),
        )
        fused2z_vd = _fused2z_all[:n_vd]
        fused2z_vn = _fused2z_all[n_vd:n_vd + n_vn]
        fused2z_te = _fused2z_all[n_vd + n_vn:]
    else:
        fused2z_vd = fused2z_vn = fused2z_te = None

    # fused_lof: rank_fuse(GPclf + Maha + LOF) — adds local density signal
    if gpclf_entropy_te is not None:
        _fused_lof_all = rank_fuse(
            np.concatenate([gpclf_entropy_vd, gpclf_entropy_vn, gpclf_entropy_te]),
            np.concatenate([maha_vd,          maha_vn,          maha_te]),
            np.concatenate([lof_vd,           lof_vn,           lof_te]),
        )
        fused_lof_vd = _fused_lof_all[:n_vd]
        fused_lof_vn = _fused_lof_all[n_vd:n_vd + n_vn]
        fused_lof_te = _fused_lof_all[n_vd + n_vn:]
    else:
        fused_lof_vd = fused_lof_vn = fused_lof_te = None

    # fused_lofz: zscore_fuse(GPclf + Maha + LOF) — best candidate for Run 6
    if gpclf_entropy_te is not None:
        _fused_lofz_all = zscore_fuse(
            np.concatenate([gpclf_entropy_vd, gpclf_entropy_vn, gpclf_entropy_te]),
            np.concatenate([maha_vd,          maha_vn,          maha_te]),
            np.concatenate([lof_vd,           lof_vn,           lof_te]),
        )
        fused_lofz_vd = _fused_lofz_all[:n_vd]
        fused_lofz_vn = _fused_lofz_all[n_vd:n_vd + n_vn]
        fused_lofz_te = _fused_lofz_all[n_vd + n_vn:]
    else:
        fused_lofz_vd = fused_lofz_vn = fused_lofz_te = None

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

        # ── LOF method (local outlier factor) ─────────────────────────────
        "lof_val_scores":    _val(lof_vd, lof_vn),
        "lof_known_scores":  lof_vd.tolist(),
        "lof_test_scores":   lof_te.tolist(),

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
        "variance_calibration_T":   T_cal,

        # ── Proposal §7: embedding distance (shift analysis) ──────────────
        "embedding_dist_test_to_train": emb_dist,
    }

    # Prototype distance (optional, requires type labels)
    if proto_te is not None:
        result.update({
            "proto_val_scores":    _val(proto_vd, proto_vn),
            "proto_known_scores":  proto_vd.tolist(),
            "proto_test_scores":   proto_te.tolist(),
        })

    # Enhancement B: GP type classifier + fused combinations (optional)
    if gpclf_entropy_te is not None:
        result.update({
            "gpclf_val_scores":    _val(gpclf_entropy_vd, gpclf_entropy_vn),
            "gpclf_known_scores":  gpclf_entropy_vd.tolist(),
            "gpclf_test_scores":   gpclf_entropy_te.tolist(),
            "fused5_val_scores":   _val(fused5_vd, fused5_vn),
            "fused5_known_scores": fused5_vd.tolist(),
            "fused5_test_scores":  fused5_te.tolist(),
        })
        if fused3_te is not None:
            result.update({
                "fused3_val_scores":   _val(fused3_vd, fused3_vn),
                "fused3_known_scores": fused3_vd.tolist(),
                "fused3_test_scores":  fused3_te.tolist(),
            })
        if fused2z_te is not None:
            result.update({
                "fused2z_val_scores":   _val(fused2z_vd, fused2z_vn),
                "fused2z_known_scores": fused2z_vd.tolist(),
                "fused2z_test_scores":  fused2z_te.tolist(),
            })
        if fused_lof_te is not None:
            result.update({
                "fused_lof_val_scores":   _val(fused_lof_vd, fused_lof_vn),
                "fused_lof_known_scores": fused_lof_vd.tolist(),
                "fused_lof_test_scores":  fused_lof_te.tolist(),
            })
        if fused_lofz_te is not None:
            result.update({
                "fused_lofz_val_scores":   _val(fused_lofz_vd, fused_lofz_vn),
                "fused_lofz_known_scores": fused_lofz_vd.tolist(),
                "fused_lofz_test_scores":  fused_lofz_te.tolist(),
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
            # Skip only if all keys (including Run 6 additions) are present
            if all(k in cached for k in (
                "fused_test_scores", "gpclf_test_scores",
                "proto_test_scores", "variance_calibration_T",
                "embedding_dist_test_to_train",
                "lof_test_scores", "fused_lofz_test_scores",
            )):
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

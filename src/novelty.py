"""Novelty scoring: GP posterior variance as out-of-distribution signal."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import GP_RESULTS_DIR, PCA_DIMS, EMBEDDINGS_DIR
from .embeddings import load_embeddings
from .folds import load_folds
from .gp_model import FittedGP, fit_gp, predict_gp
from .pca_transform import FoldPCA, fit_fold_pca

log = logging.getLogger(__name__)


def run_single_fold(
    fold: dict[str, Any],
    pca_dim: int,
    device: Optional[str] = None,
    save_pca: bool = True,
    pca_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Full GP pipeline for one LODTO fold:
      1. Load cached DINOv2 embeddings
      2. Fit per-fold PCA on train_defective only
      3. Fit ExactGP on (X_train_pca, log(severity))
      4. Score val_defective, val_normal, test_defective with posterior variance

    Returns a serialisable result dict.
    """
    train_paths    = fold["train_defective"]
    val_def_paths  = fold["val_defective"]
    val_norm_paths = fold["val_normal"]
    test_paths     = fold["test_defective"]

    # 1. Embeddings
    X_train    = load_embeddings(train_paths)
    X_val_def  = load_embeddings(val_def_paths)
    X_val_norm = load_embeddings(val_norm_paths)
    X_test     = load_embeddings(test_paths)

    # 2. PCA (fitted on training defectives)
    fpca = fit_fold_pca(X_train, pca_dim)
    X_tr  = fpca.transform(X_train)
    X_vd  = fpca.transform(X_val_def)
    X_vn  = fpca.transform(X_val_norm)
    X_te  = fpca.transform(X_test)

    if save_pca and pca_dir is not None:
        pca_path = pca_dir / f"{fold['fold_id']}_pca.pkl"
        fpca.save(pca_path)

    # 3. GP on log(severity)
    y_train = np.log(np.array(fold["train_severity"], dtype=np.float64))
    fitted  = fit_gp(X_tr.astype(np.float64), y_train, device=device)

    # 4. Posterior variance (novelty score — high = likely novel)
    _, var_val_def  = predict_gp(X_vd.astype(np.float64), fitted)
    _, var_val_norm = predict_gp(X_vn.astype(np.float64), fitted)
    _, var_test     = predict_gp(X_te.astype(np.float64), fitted)

    # Combine val splits for threshold optimisation
    val_scores = np.concatenate([var_val_def, var_val_norm])
    val_labels = np.concatenate([
        np.ones(len(var_val_def)),    # novel defective = positive
        np.zeros(len(var_val_norm)),  # normal = negative
    ])

    # GP variance on val_defective (known) — used for defect-only AUROC
    _, var_val_known = predict_gp(X_vd.astype(np.float64), fitted)

    return {
        "fold_id":              fold["fold_id"],
        "category":             fold["category"],
        "held_out_type":        fold["held_out_type"],
        "pca_dim":              pca_dim,
        "kernel_name":          fitted.kernel_name,
        "lml":                  float(fitted.lml),
        "pca_var_explained":    float(fpca.explained_variance_ratio.sum()),
        # validation (for threshold tuning)
        "val_scores":           val_scores.tolist(),
        "val_labels":           val_labels.tolist(),
        # known defective scores (for defect-only AUROC)
        "known_scores":         var_val_known.tolist(),
        # test (held-out novel type — primary evaluation)
        "test_scores":          var_test.tolist(),
        "test_severity":        fold["test_severity"],
        "test_defect_types":    fold["test_defect_types"],
        # counts
        "n_train":              len(train_paths),
        "n_test":               len(test_paths),
    }


def run_all_folds(
    pca_dim: int = 16,
    device: Optional[str] = None,
    save_dir: Path = GP_RESULTS_DIR,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """
    Run all LODTO folds for a given PCA dimension.
    Results saved per-fold as JSON; resume=True skips cached folds.
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
            log.info(f"[{i+1}/{len(folds)}] Skip {fid} (cached)")
            with open(out_path) as f:
                results.append(json.load(f))
            continue

        log.info(f"[{i+1}/{len(folds)}] {fid}  pca_dim={pca_dim}")
        result = run_single_fold(fold, pca_dim, device=device, save_pca=True, pca_dir=pca_dir)

        with open(out_path, "w") as f:
            json.dump(result, f)
        results.append(result)

    log.info(f"GP complete: {len(results)} folds, pca_dim={pca_dim}.")
    return results

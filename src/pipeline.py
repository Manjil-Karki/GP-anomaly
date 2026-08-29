"""End-to-end pipeline orchestration (Phases 0–6)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import (
    DATA_ROOT, MANIFEST_PATH, FOLDS_PATH,
    EMBEDDINGS_DIR, GP_RESULTS_DIR, BASELINE_RESULTS_DIR, EVAL_DIR,
    PCA_DIMS, RANDOM_SEED,
)
from .data import build_manifest, load_manifest
from .embeddings import cache_all_embeddings, load_dinov2, load_embeddings
from .evaluation import (
    holm_correct, save_json, score_fold,
    build_fold_table, summarise, wilcoxon_test,
)
from .folds import (
    build_lodto_folds, check_leakage, fold_summary, load_folds, save_folds,
)
from .novelty import run_all_folds
from .baselines import run_baseline_folds
from .visualize import run_method_figures, plot_pca_dim_sweep

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def phase0_manifest(force: bool = False) -> pd.DataFrame:
    """Build or reload the dataset manifest."""
    if not force and MANIFEST_PATH.exists():
        log.info("Phase 0: loading existing manifest.")
        return load_manifest()
    log.info("Phase 0: building manifest from MVTec AD dataset…")
    df = build_manifest(DATA_ROOT)
    log.info(f"  {len(df)} images catalogued.")
    return df


def phase1_folds(df: pd.DataFrame, force: bool = False) -> list[dict]:
    """Construct LODTO folds and run leakage check."""
    if not force and FOLDS_PATH.exists():
        log.info("Phase 1: loading existing folds.")
        folds = load_folds()
    else:
        log.info("Phase 1: building LODTO folds…")
        folds = build_lodto_folds(df)
        save_folds(folds)
        log.info(f"  {len(folds)} folds written to {FOLDS_PATH}")

    violations = check_leakage(folds)
    if violations:
        raise RuntimeError("Data leakage detected:\n" + "\n".join(violations))
    log.info(f"Phase 1: leakage check passed ({len(folds)} folds).")

    summary = fold_summary(folds)
    log.info(f"\n{summary.to_string(index=False)}")
    return folds


def phase2_embeddings(df: pd.DataFrame, device: Optional[str] = None, force: bool = False) -> None:
    """Cache DINOv2 ViT-B/14 embeddings for all images."""
    index_path = EMBEDDINGS_DIR / "index.json"
    if not force and index_path.exists():
        with open(index_path) as f:
            cached = json.load(f)
        if len(cached) >= len(df):
            log.info(f"Phase 2: embeddings already cached ({len(cached)} images).")
            return

    log.info("Phase 2: loading DINOv2 backbone…")
    model, dev = load_dinov2(device)
    log.info(f"  Extracting embeddings on {dev}…")
    cache_all_embeddings(df["image_path"].tolist(), model, dev)
    log.info("Phase 2: embedding cache complete.")


def phase3_gp(
    pca_dim: int = 16,
    device: Optional[str] = None,
    resume: bool = True,
) -> list[dict]:
    """Fit ExactGP on all 72 LODTO folds."""
    log.info(f"Phase 3: GP inference (pca_dim={pca_dim})…")
    results = run_all_folds(pca_dim=pca_dim, device=device, save_dir=GP_RESULTS_DIR, resume=resume)
    save_json(results, f"gp_pca{pca_dim}", EVAL_DIR)

    # Method figures (uncertainty surface, kernel selection)
    try:
        folds = load_folds()
        run_method_figures(results, folds, device=device)
    except Exception as e:
        log.warning(f"Method figures skipped: {e}")

    return results


def phase4_threshold(
    gp_results: list[dict],
    df: pd.DataFrame,
) -> list[dict]:
    """Optimise per-fold threshold and compute AUROC metrics."""
    log.info("Phase 4: threshold optimisation and scoring…")
    scored = []
    for result in gp_results:
        val_scores    = np.array(result["val_scores"])
        val_labels    = np.array(result["val_labels"])
        normal_scores = val_scores[val_labels == 0]

        # known_scores stored directly in fold result (GP variance on val_defective)
        known_scores = np.array(result.get("known_scores", []))

        fold_score = score_fold(result, known_scores, normal_scores)
        scored.append(fold_score)

    save_json(scored, "scored_folds", EVAL_DIR)
    log.info(f"  Mean detection rate: {np.nanmean([s['detection_rate'] for s in scored]):.3f}")
    return scored


def phase5_baselines(folds: list[dict], device: Optional[str] = None, resume: bool = True) -> dict:
    """Run deep ensemble and MC dropout baselines."""
    log.info("Phase 5: baselines…")
    results = run_baseline_folds(folds, device=device, resume=resume)
    for name, res in results.items():
        save_json(res, f"baseline_{name}", EVAL_DIR)
    return results


def phase6_evaluation(
    scored_folds: list[dict],
    baseline_results: dict[str, list[dict]],
) -> dict:
    """Wilcoxon + Holm tests, summary statistics, final table."""
    log.info("Phase 6: statistical evaluation…")

    fold_df = build_fold_table(scored_folds)
    summary = summarise(fold_df)
    gp_auroc = fold_df["auroc_do"].values

    # --- compute per-fold baseline AUROC (defect-only) ---
    baseline_aurocs: dict[str, np.ndarray] = {}
    for name, b_results in baseline_results.items():
        aucs = []
        for r in b_results:
            novel_sc = np.array(r["test_scores"])
            known_sc = np.array(r.get("known_scores", []))
            if len(known_sc) == 0:
                aucs.append(float("nan"))
                continue
            scores = np.concatenate([novel_sc, known_sc])
            labels = np.concatenate([np.ones(len(novel_sc)), np.zeros(len(known_sc))])
            from sklearn.metrics import roc_auc_score
            try:
                aucs.append(float(roc_auc_score(labels, scores)))
            except Exception:
                aucs.append(float("nan"))
        baseline_aurocs[name] = np.array(aucs)
        b_df = pd.DataFrame(b_results)
        b_df["auroc_do"] = aucs
        b_df.to_csv(EVAL_DIR / f"baseline_{name}_fold_results.csv", index=False)

    # --- statistical tests ---
    tests = []
    for name, b_aucs in baseline_aurocs.items():
        tests.append(wilcoxon_test(gp_auroc, b_aucs, name))
    tests = holm_correct(tests)

    # --- PCA dim sweep (if multiple dims were run) ---
    all_dim_results: dict[int, list[dict]] = {}
    for d in [8, 12, 16]:
        p = EVAL_DIR / f"gp_pca{d}.json"
        if p.exists():
            with open(p) as f:
                all_dim_results[d] = json.load(f)
    if len(all_dim_results) > 1:
        try:
            from .visualize import plot_pca_dim_sweep
            plot_pca_dim_sweep(all_dim_results)
        except Exception as e:
            log.warning(f"PCA sweep figure skipped: {e}")

    eval_summary = {
        "n_folds":             int(len(scored_folds)),
        "mean_auroc_do":       float(np.nanmean(gp_auroc)),
        "std_auroc_do":        float(np.nanstd(gp_auroc)),
        "mean_detection_rate": float(np.nanmean(fold_df["detection_rate"])),
        "category_summary":    summary.to_dict(orient="records"),
        "wilcoxon_tests":      tests,
        "baseline_mean_auroc": {
            k: float(np.nanmean(v)) for k, v in baseline_aurocs.items()
        },
    }

    save_json(eval_summary, "evaluation_summary", EVAL_DIR)
    fold_df.to_csv(EVAL_DIR / "fold_results.csv", index=False)
    summary.to_csv(EVAL_DIR / "category_summary.csv", index=False)

    log.info(f"\nOverall AUROC (defect-only): {eval_summary['mean_auroc_do']:.3f} ± {eval_summary['std_auroc_do']:.3f}")
    for name, b_aucs in baseline_aurocs.items():
        log.info(f"  {name}: {np.nanmean(b_aucs):.3f}")
    log.info(f"Statistical tests: {tests}")
    return eval_summary


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_pipeline(
    phases: Optional[list[int]] = None,
    pca_dim: int = 16,
    device: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Run end-to-end pipeline.

    phases — list of phase numbers 0–6 (default: all).
      0: build manifest
      1: construct LODTO folds
      2: extract DINOv2 embeddings
      3: GP inference
      4: threshold optimisation + AUROC
      5: baselines
      6: statistical evaluation + final tables
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    if phases is None:
        phases = list(range(7))

    df     = None
    folds  = None
    gp_res = None
    bl_res = None
    scored = None

    if 0 in phases:
        df = phase0_manifest(force=force)
    if df is None:
        df = load_manifest()

    if 1 in phases:
        folds = phase1_folds(df, force=force)
    if folds is None and FOLDS_PATH.exists():
        folds = load_folds()

    if 2 in phases:
        phase2_embeddings(df, device=device, force=force)

    if 3 in phases:
        gp_res = phase3_gp(pca_dim=pca_dim, device=device)

    if 4 in phases:
        if gp_res is None:
            p = EVAL_DIR / f"gp_pca{pca_dim}.json"
            with open(p) as f:
                gp_res = json.load(f)
        scored = phase4_threshold(gp_res, df)

    if 5 in phases:
        if folds is None:
            raise RuntimeError("Phase 5 requires folds. Run phase 1 first.")
        bl_res = phase5_baselines(folds, device=device)

    if 6 in phases:
        if scored is None:
            with open(EVAL_DIR / "scored_folds.json") as f:
                scored = json.load(f)
        if bl_res is None:
            bl_res = {}
            for name in ("ensemble", "mc_dropout"):
                p = EVAL_DIR / f"baseline_{name}.json"
                if p.exists():
                    with open(p) as f:
                        bl_res[name] = json.load(f)
        phase6_evaluation(scored, bl_res)

    log.info("Pipeline complete.")

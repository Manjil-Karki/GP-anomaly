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
    holm_correct, save_json, score_fold, score_all_methods,
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
        run_method_figures(results, folds, device=device, pca_dim=pca_dim)
    except Exception as e:
        log.warning(f"Method figures skipped: {e}")

    return results


def _emit_cost_curve(
    gp_results: list[dict],
    all_method: dict[str, list[dict]],
) -> None:
    """Save C3_cost_curve.png for the median-AUROC GP fold."""
    import matplotlib.pyplot as plt
    from .config import FIGURES_DIR
    from .threshold import cost_sweep as _cs

    gp_folds = all_method.get("gp", [])
    if not gp_folds:
        return

    aurocs  = [s["auroc_do"] for s in gp_folds]
    med_idx = int(np.argsort(aurocs)[len(aurocs) // 2])
    rep     = gp_results[med_idx]

    val_scores = np.array(rep["val_scores"])
    val_labels = np.array(rep["val_labels"])
    sweep      = _cs(val_scores, val_labels)
    save_json(sweep, "cost_curve_sweep", EVAL_DIR)

    ratios = [r["c_fn_c_fp_ratio"] for r in sweep]
    miss   = [r["miss_rate"] for r in sweep]
    alarm  = [r["false_alarm_rate"] for r in sweep]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ratios, miss,  "o-",  label="Miss rate  P(FN)")
    ax.plot(ratios, alarm, "s--", label="False-alarm rate  P(FP)")
    ax.set_xlabel("$C_{FN}/C_{FP}$ ratio")
    ax.set_ylabel("Rate")
    ax.set_title("Cost-Optimal Operating Points\n"
                 f"(median-AUROC fold: {rep['fold_id']})")
    ax.legend()
    ax.set_xscale("log")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "C3_cost_curve.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out.name}")


def phase4_threshold(
    gp_results: list[dict],
    df: pd.DataFrame,
) -> list[dict]:
    """
    Threshold optimisation + AUROC/AUPR for all 4 novelty methods.
    Also runs GP calibration and generates the cost-operating curve.
    """
    from .calibration import run_calibration

    log.info("Phase 4: threshold optimisation and multi-method scoring…")

    gp_scored: list[dict] = []
    all_method: dict[str, list[dict]] = {}

    for result in gp_results:
        val_scores   = np.array(result["val_scores"])
        val_labels   = np.array(result["val_labels"])
        normal_sc    = val_scores[val_labels == 0]
        known_sc     = np.array(result.get("known_scores", []))

        # GP fold score (backward-compat key "scored_folds")
        gp_scored.append(score_fold(result, known_sc, normal_sc))

        # All 4 methods
        for method, ms in score_all_methods(result).items():
            all_method.setdefault(method, []).append(ms)

    save_json(gp_scored, "scored_folds", EVAL_DIR)
    for method, scores in all_method.items():
        save_json(scores, f"scored_folds_{method}", EVAL_DIR)

    log.info("Phase 4: GP calibration…")
    try:
        run_calibration(gp_results, save_dir=EVAL_DIR)
    except Exception as e:
        log.warning(f"Calibration skipped: {e}")

    log.info("Phase 4: cost curve sweep…")
    try:
        _emit_cost_curve(gp_results, all_method)
    except Exception as e:
        log.warning(f"Cost curve skipped: {e}")

    log.info(f"  GP mean detection rate: {np.nanmean([s['detection_rate'] for s in gp_scored]):.3f}")
    return gp_scored


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
    """
    Wilcoxon + Holm tests across all novelty methods and baselines.
    Loads per-method scored folds from EVAL_DIR if available (written by phase 4).
    """
    log.info("Phase 6: statistical evaluation…")

    fold_df  = build_fold_table(scored_folds)
    summary  = summarise(fold_df)
    gp_auroc = fold_df["auroc_do"].values

    # --- load per-method AUROC / AUPR from phase 4 outputs ---
    method_aurocs: dict[str, np.ndarray] = {"gp": gp_auroc}
    method_aupr:   dict[str, np.ndarray] = {}
    method_dr:     dict[str, np.ndarray] = {}

    for method in ("gp", "knn", "maha", "lof", "fused", "gpclf", "proto",
                   "fused2", "fused2z", "fused3", "fused5", "fused_lof", "fused_lofz"):
        p = EVAL_DIR / f"scored_folds_{method}.json"
        if not p.exists():
            continue
        with open(p) as f:
            mdf = pd.DataFrame(json.load(f))
        method_aurocs[method] = mdf["auroc_do"].values
        if "aupr_do" in mdf.columns:
            method_aupr[method] = mdf["aupr_do"].values
        if "detection_rate" in mdf.columns:
            method_dr[method] = mdf["detection_rate"].values

    # --- baseline AUROC (defect-only) ---
    baseline_aurocs: dict[str, np.ndarray] = {}
    for name, b_results in baseline_results.items():
        from sklearn.metrics import roc_auc_score
        aucs = []
        for r in b_results:
            novel_sc = np.array(r["test_scores"])
            known_sc = np.array(r.get("known_scores", []))
            if len(known_sc) == 0:
                aucs.append(float("nan"))
                continue
            scores = np.concatenate([novel_sc, known_sc])
            labels = np.concatenate([np.ones(len(novel_sc)), np.zeros(len(known_sc))])
            try:
                aucs.append(float(roc_auc_score(labels, scores)))
            except Exception:
                aucs.append(float("nan"))
        baseline_aurocs[name] = np.array(aucs)
        b_df = pd.DataFrame(b_results)
        b_df["auroc_do"] = aucs
        b_df.to_csv(EVAL_DIR / f"baseline_{name}_fold_results.csv", index=False)

    # --- statistical tests: reference = best fusion method available ---
    for candidate in ("fused_lofz", "fused_lof", "fused2z", "fused2", "fused3", "fused5", "fused", "gp"):
        if candidate in method_aurocs:
            ref_name = candidate
            break
    else:
        ref_name = "gp"
    ref_auroc = method_aurocs[ref_name]

    tests = []
    # our methods vs each other
    for m in ("gp", "knn", "maha", "lof", "gpclf", "proto",
              "fused", "fused2", "fused2z", "fused3", "fused5", "fused_lof", "fused_lofz"):
        if m == ref_name or m not in method_aurocs:
            continue
        tests.append(wilcoxon_test(ref_auroc, method_aurocs[m], f"{ref_name}_vs_{m}"))
    # our best vs baselines
    for name, b_aucs in baseline_aurocs.items():
        tests.append(wilcoxon_test(ref_auroc, b_aucs, f"{ref_name}_vs_{name}"))
    # also GP vs baselines (legacy / proposal comparison)
    if ref_name != "gp":
        for name, b_aucs in baseline_aurocs.items():
            tests.append(wilcoxon_test(gp_auroc, b_aucs, f"gp_vs_{name}"))
    tests = holm_correct(tests)

    # --- PCA dim sweep figure (load all gp_pca*.json files present) ---
    import re as _re
    all_dim_results: dict[int, list[dict]] = {}
    for p in sorted(EVAL_DIR.glob("gp_pca*.json")):
        m = _re.search(r"gp_pca(\d+)\.json$", p.name)
        if m:
            with open(p) as f:
                all_dim_results[int(m.group(1))] = json.load(f)
    if len(all_dim_results) > 1:
        try:
            plot_pca_dim_sweep(all_dim_results)
        except Exception as e:
            log.warning(f"PCA sweep figure skipped: {e}")

    # --- method comparison table ---
    comparison_rows = []
    for method, aucs in method_aurocs.items():
        row: dict = {
            "method":        method,
            "mean_auroc_do": float(np.nanmean(aucs)),
            "std_auroc_do":  float(np.nanstd(aucs)),
        }
        if method in method_aupr:
            row["mean_aupr_do"] = float(np.nanmean(method_aupr[method]))
        if method in method_dr:
            row["mean_detection_rate"] = float(np.nanmean(method_dr[method]))
        comparison_rows.append(row)
    for name, aucs in baseline_aurocs.items():
        comparison_rows.append({
            "method":        f"baseline_{name}",
            "mean_auroc_do": float(np.nanmean(aucs)),
            "std_auroc_do":  float(np.nanstd(aucs)),
        })
    pd.DataFrame(comparison_rows).sort_values(
        "mean_auroc_do", ascending=False
    ).to_csv(EVAL_DIR / "method_comparison.csv", index=False)

    eval_summary = {
        "n_folds":             int(len(scored_folds)),
        "mean_auroc_do":       float(np.nanmean(gp_auroc)),
        "std_auroc_do":        float(np.nanstd(gp_auroc)),
        "mean_detection_rate": float(np.nanmean(fold_df["detection_rate"])),
        "category_summary":    summary.to_dict(orient="records"),
        "wilcoxon_tests":      tests,
        "method_comparison":   comparison_rows,
        "baseline_mean_auroc": {k: float(np.nanmean(v)) for k, v in baseline_aurocs.items()},
    }

    save_json(eval_summary, "evaluation_summary", EVAL_DIR)
    fold_df.to_csv(EVAL_DIR / "fold_results.csv", index=False)
    summary.to_csv(EVAL_DIR / "category_summary.csv", index=False)

    log.info("\nMethod comparison (AUROC defect-only):")
    for row in sorted(comparison_rows, key=lambda r: r["mean_auroc_do"], reverse=True):
        aupr_s = f"  AUPR={row['mean_aupr_do']:.3f}" if "mean_aupr_do" in row else ""
        log.info(f"  {row['method']:20s}  {row['mean_auroc_do']:.3f} ± {row['std_auroc_do']:.3f}{aupr_s}")
    log.info(f"Statistical tests (Holm-corrected): {[t['method'] for t in tests]}")
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
        bl_res = phase5_baselines(folds, device=device, resume=not force)

    if 6 in phases:
        if scored is None:
            with open(EVAL_DIR / "scored_folds.json") as f:
                scored = json.load(f)
        if bl_res is None:
            bl_res = {}
            for name in ("ensemble", "mc_dropout", "padim", "patchcore"):
                p = EVAL_DIR / f"baseline_{name}.json"
                if p.exists():
                    with open(p) as f:
                        bl_res[name] = json.load(f)
        phase6_evaluation(scored, bl_res)

    log.info("Pipeline complete.")

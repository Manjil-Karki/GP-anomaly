"""LODTO fold construction and leakage verification.

MVTec AD structure: ALL defective images are in the test split.
Train split contains normal (good) images only.

Fold layout:
  train_defective  — test-split defectives of all types EXCEPT held_out (train portion)
  val_defective    — same pool, validation portion (for threshold tuning)
  test_defective   — test-split defectives of the held_out type (novelty probe)
  train_normal     — train-split normals (train portion, for baselines / context)
  val_normal       — train-split normals (validation portion)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FOLDS_PATH, LODTO_CATEGORIES, RANDOM_SEED


def build_lodto_folds(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
) -> list[dict[str, Any]]:
    """
    Leave-One-Defect-Type-Out folds across 14 LODTO categories.

    For each (category, held_out_defect_type):
      - test:           all test-split images of the held-out type (novel probe)
      - train_defective: known test-split defectives (other types), train portion
      - val_defective:   known test-split defectives, val portion
      - train_normal:    train-split normals, train portion
      - val_normal:      train-split normals, val portion
    """
    rng = np.random.default_rng(RANDOM_SEED)
    folds: list[dict[str, Any]] = []

    for category in LODTO_CATEGORIES:
        cat_df = df[df["category"] == category].copy()
        defect_types = [t for t in cat_df["defect_type"].unique() if t != "good"]
        if len(defect_types) < 2:
            continue  # need ≥2 types for LODTO

        # Normal images (train split only)
        normals = cat_df[
            (cat_df["defect_type"] == "good") & (cat_df["split"] == "train")
        ]["image_path"].tolist()
        rng.shuffle(normals)
        n_val_n = max(1, int(len(normals) * val_fraction))
        val_normal   = normals[:n_val_n]
        train_normal = normals[n_val_n:]

        for held_out in defect_types:
            # Novel test set: held-out type in test split
            test_rows = cat_df[
                (cat_df["defect_type"] == held_out) & (cat_df["split"] == "test")
            ]

            # Known defectives: all OTHER defect types in test split
            known_def = cat_df[
                (cat_df["defect_type"] != held_out)
                & (cat_df["defect_type"] != "good")
                & (cat_df["split"] == "test")
            ]

            def_paths = known_def["image_path"].tolist()
            def_sev   = known_def["severity"].tolist()

            idx = rng.permutation(len(def_paths)).tolist()
            def_paths = [def_paths[i] for i in idx]
            def_sev   = [def_sev[i]   for i in idx]

            n_val_d = max(1, int(len(def_paths) * val_fraction))
            val_def_paths  = def_paths[:n_val_d]
            val_def_sev    = def_sev[:n_val_d]
            train_def_paths = def_paths[n_val_d:]
            train_def_sev   = def_sev[n_val_d:]

            folds.append({
                "fold_id":           f"{category}__{held_out}",
                "category":          category,
                "held_out_type":     held_out,
                # GP training set (known defectives)
                "train_defective":   train_def_paths,
                "train_severity":    train_def_sev,
                "train_normal":      train_normal,
                # Validation set (threshold tuning / PCA dim selection)
                "val_defective":     val_def_paths,
                "val_severity":      val_def_sev,
                "val_normal":        val_normal,
                # Test set: novel defect type (the main evaluation)
                "test_defective":    test_rows["image_path"].tolist(),
                "test_defect_types": test_rows["defect_type"].tolist(),
                "test_severity":     test_rows["severity"].tolist(),
                # Counts
                "n_train_defective": len(train_def_paths),
                "n_val_defective":   len(val_def_paths),
                "n_test":            len(test_rows),
                "n_train_normal":    len(train_normal),
                "n_val_normal":      len(val_normal),
            })

    return folds


def save_folds(folds: list[dict], path: Path = FOLDS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(folds, f, indent=2)


def load_folds(path: Path = FOLDS_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def check_leakage(folds: list[dict]) -> list[str]:
    """Return list of violation strings. Empty → no leakage."""
    violations = []
    for fold in folds:
        fid = fold["fold_id"]
        ht  = fold["held_out_type"]
        s_tr_def  = set(fold["train_defective"])
        s_va_def  = set(fold["val_defective"])
        s_test    = set(fold["test_defective"])
        s_tr_norm = set(fold["train_normal"])
        s_va_norm = set(fold["val_normal"])

        if s_tr_def & s_va_def:
            violations.append(f"{fid}: train_defective ∩ val_defective")
        if s_tr_def & s_test:
            violations.append(f"{fid}: train_defective ∩ test")
        if s_va_def & s_test:
            violations.append(f"{fid}: val_defective ∩ test")
        if s_tr_norm & s_va_norm:
            violations.append(f"{fid}: train_normal ∩ val_normal")

        # Held-out type must not appear in any training path
        for p in fold["train_defective"]:
            if f"/{ht}/" in p:
                violations.append(f"{fid}: held-out '{ht}' found in train_defective path")
                break

    return violations


def fold_summary(folds: list[dict]) -> "pd.DataFrame":
    import pandas as pd
    rows = [{
        "fold_id":           f["fold_id"],
        "category":          f["category"],
        "held_out_type":     f["held_out_type"],
        "n_train_defective": f["n_train_defective"],
        "n_val_defective":   f["n_val_defective"],
        "n_test":            f["n_test"],
        "n_train_normal":    f["n_train_normal"],
    } for f in folds]
    return pd.DataFrame(rows)

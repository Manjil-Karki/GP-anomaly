"""Baseline anomaly detectors: deep ensemble, MC dropout, PaDiM, PatchCore.

torch is imported lazily so phases 0-1 work without the GPU stack.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import (
    ENSEMBLE_SIZE, MC_DROPOUT_PASSES, MC_DROPOUT_RATE,
    MLP_BATCH_SIZE, MLP_EPOCHS, MLP_HIDDEN, RANDOM_SEED,
    BASELINE_RESULTS_DIR,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLP autoencoder (defined lazily when torch is available)
# ---------------------------------------------------------------------------

def _make_mlp(in_dim: int, hidden: list[int], dropout: float = 0.0):
    import torch.nn as nn

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            enc, d = [], in_dim
            for h in hidden:
                enc += [nn.Linear(d, h), nn.ReLU()]
                if dropout > 0:
                    enc.append(nn.Dropout(dropout))
                d = h
            bottleneck = max(hidden[-1] // 2, 8)
            enc.append(nn.Linear(d, bottleneck))
            self.encoder = nn.Sequential(*enc)

            dec, d = [], bottleneck
            for h in reversed(hidden):
                dec += [nn.Linear(d, h), nn.ReLU()]
                d = h
            dec.append(nn.Linear(d, in_dim))
            self.decoder = nn.Sequential(*dec)

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return _MLP()


def _train(X: np.ndarray, dropout: float, seed: int, device: str):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    model = _make_mlp(X.shape[1], MLP_HIDDEN, dropout).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds    = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dl    = DataLoader(ds, batch_size=MLP_BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(MLP_EPOCHS):
        for (batch,) in dl:
            batch = batch.to(device)
            loss  = nn.functional.mse_loss(model(batch), batch)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def _recon_error(model, X: np.ndarray, device: str, train_mode=False) -> np.ndarray:
    import torch
    import torch.nn as nn
    if train_mode:
        model.train()
    else:
        model.eval()
    x_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        err = nn.functional.mse_loss(model(x_t), x_t, reduction="none").mean(1)
    return err.cpu().numpy()


# ---------------------------------------------------------------------------
# Deep ensemble
# ---------------------------------------------------------------------------

def deep_ensemble_scores(
    X_train_normal: np.ndarray,
    X_test: np.ndarray,
    device: Optional[str] = None,
    n_members: int = ENSEMBLE_SIZE,
) -> np.ndarray:
    """Mean reconstruction error over n_members independently trained autoencoders."""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    errors = np.stack([
        _recon_error(
            _train(X_train_normal, 0.0, RANDOM_SEED + m, device),
            X_test, device
        )
        for m in range(n_members)
    ])
    return errors.mean(0)


# ---------------------------------------------------------------------------
# MC dropout
# ---------------------------------------------------------------------------

def mc_dropout_scores(
    X_train_normal: np.ndarray,
    X_test: np.ndarray,
    device: Optional[str] = None,
    n_passes: int = MC_DROPOUT_PASSES,
    dropout: float = MC_DROPOUT_RATE,
) -> np.ndarray:
    """Mean reconstruction error over n stochastic dropout forward passes."""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _train(X_train_normal, dropout, RANDOM_SEED, device)
    errors = np.stack([
        _recon_error(model, X_test, device, train_mode=True)
        for _ in range(n_passes)
    ])
    return errors.mean(0)


# ---------------------------------------------------------------------------
# PaDiM-like and PatchCore-like baselines (embedding-level)
#
# Both train on NORMAL images and detect deviation from normality.
# This is the standard anomaly detection paradigm, unlike our GP which trains
# on known defect types. These serve as reference comparisons showing what a
# normal-distribution baseline can achieve on the LODTO task.
#
# PaDiM-like: fit a multivariate Gaussian on PCA-compressed normal embeddings,
#   score = Mahalanobis distance from the normal distribution.
#   (PaDiM: Defard et al. 2021 — our version operates on mean patch tokens
#   rather than per-position patch features due to storage constraints.)
#
# PatchCore-like: nearest-neighbour distance from a coreset of normal
#   embeddings.  (PatchCore: Roth et al. CVPR 2022 — our version uses mean
#   patch tokens as the memory bank entries.)
# ---------------------------------------------------------------------------

def _pca_reduce(X_train: np.ndarray, X_test: np.ndarray, n_components: int = 32):
    """Fit PCA on training data, apply to both. Returns (X_tr_pca, X_te_pca)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    n_comp = min(n_components, X_tr.shape[0] - 1, X_tr.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(scaler.transform(X_test))
    return X_tr_pca, X_te_pca


def padim_scores(
    X_train_normal: np.ndarray,
    X_test: np.ndarray,
    pca_dim: int = 32,
) -> np.ndarray:
    """
    PaDiM-like novelty score: Mahalanobis distance from the normal-image
    distribution in PCA-reduced embedding space.
    """
    from .novelty_scores import mahalanobis_score
    X_tr, X_te = _pca_reduce(X_train_normal, X_test, pca_dim)
    return mahalanobis_score(X_te, X_tr)


def patchcore_scores(
    X_train_normal: np.ndarray,
    X_test: np.ndarray,
    k: int = 1,
) -> np.ndarray:
    """
    PatchCore-like novelty score: nearest-neighbour distance from the normal-
    image memory bank in the full embedding space.
    """
    from .novelty_scores import knn_score
    return knn_score(X_test, X_train_normal, k=k)


# ---------------------------------------------------------------------------
# Run baselines for all folds
# ---------------------------------------------------------------------------

def run_baseline_folds(
    folds: list[dict[str, Any]],
    device: Optional[str] = None,
    resume: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Run deep ensemble and MC dropout on all folds with per-fold JSON caching."""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    BASELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {
        "ensemble": [], "mc_dropout": [], "padim": [], "patchcore": []
    }

    for name in ("ensemble", "mc_dropout", "padim", "patchcore"):
        out_dir = BASELINE_RESULTS_DIR / name
        out_dir.mkdir(exist_ok=True)

        for i, fold in enumerate(folds):
            fid      = fold["fold_id"]
            out_path = out_dir / f"{fid}.json"

            if resume and out_path.exists():
                log.info(f"[{name}] Skip {fid} (cached)")
                with open(out_path) as f:
                    results[name].append(json.load(f))
                continue

            log.info(f"[{i+1}/{len(folds)}] {name} — {fid}")
            from .embeddings import load_embeddings
            X_norm  = load_embeddings(fold["train_normal"])
            X_test  = load_embeddings(fold["test_defective"])   # novel (label=1)
            X_known = load_embeddings(fold["val_defective"])    # known defectives (label=0)

            if name == "ensemble":
                scores_novel = deep_ensemble_scores(X_norm, X_test,  device=device)
                scores_known = deep_ensemble_scores(X_norm, X_known, device=device)
            elif name == "mc_dropout":
                scores_novel = mc_dropout_scores(X_norm, X_test,  device=device)
                scores_known = mc_dropout_scores(X_norm, X_known, device=device)
            elif name == "padim":
                scores_novel = padim_scores(X_norm, X_test)
                scores_known = padim_scores(X_norm, X_known)
            else:  # patchcore
                scores_novel = patchcore_scores(X_norm, X_test)
                scores_known = patchcore_scores(X_norm, X_known)

            record = {
                "fold_id":       fid,
                "category":      fold["category"],
                "held_out_type": fold["held_out_type"],
                "test_scores":   scores_novel.tolist(),
                "known_scores":  scores_known.tolist(),
                "n_test":        len(X_test),
            }
            with open(out_path, "w") as f:
                json.dump(record, f)
            results[name].append(record)

    return results

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
# PaDiM / PatchCore via anomalib
# ---------------------------------------------------------------------------

def _anomalib_scores(
    model_name: str,
    train_paths: list[str],
    test_paths: list[str],
    category: str,
    device: Optional[str] = None,
) -> np.ndarray:
    try:
        import anomalib  # noqa: F401
    except ImportError:
        log.warning(f"anomalib not installed; {model_name} unavailable.")
        return np.full(len(test_paths), np.nan)
    log.warning(f"{model_name}: anomalib datamodule wiring pending (Phase 7).")
    return np.full(len(test_paths), np.nan)


def padim_scores(
    train_paths: list[str],
    test_paths: list[str],
    category: str,
    device: Optional[str] = None,
) -> np.ndarray:
    return _anomalib_scores("PaDiM", train_paths, test_paths, category, device)


def patchcore_scores(
    train_paths: list[str],
    test_paths: list[str],
    category: str,
    device: Optional[str] = None,
) -> np.ndarray:
    return _anomalib_scores("PatchCore", train_paths, test_paths, category, device)


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
    results: dict[str, list[dict]] = {"ensemble": [], "mc_dropout": []}

    for name in ("ensemble", "mc_dropout"):
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
            else:
                scores_novel = mc_dropout_scores(X_norm, X_test,  device=device)
                scores_known = mc_dropout_scores(X_norm, X_known, device=device)

            record = {
                "fold_id":       fid,
                "category":      fold["category"],
                "held_out_type": fold["held_out_type"],
                "test_scores":   scores_novel.tolist(),   # novel defectives
                "known_scores":  scores_known.tolist(),   # known defectives (for AUROC)
                "n_test":        len(X_test),
            }
            with open(out_path, "w") as f:
                json.dump(record, f)
            results[name].append(record)

    return results

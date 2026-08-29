"""
Distance-based OOD novelty scores for the LODTO pipeline.

k-NN distance — Sun et al. (ICML 2022) "Out-of-Distribution Detection with
Deep Nearest Neighbors." k-th nearest neighbour distance in compressed PCA
space. Directly fixes the convex-hull failure mode: a novel type that projects
inside the training scatter but far from every cluster still has a large k-NN
distance, whereas GP variance can be low in those regions.

Mahalanobis distance — Lee et al. (NeurIPS 2018) "A Simple Unified Framework
for Detecting Out-of-Distribution Samples and Adversarial Attacks." Distance
from the training centroid corrected for covariance. Captures cross-dimension
correlations that GP ARD length-scales treat independently.

Rank fusion — rank-normalise each score to [0,1] then average. Combines
signals of different magnitudes and scales without a shared parametric
assumption. Our proposed best method: GP variance + k-NN + Mahalanobis.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_score(
    X_test: np.ndarray,
    X_train: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """
    k-th nearest neighbour distance in PCA-compressed embedding space.
    High score → test point is far from every training cluster → likely novel.
    k is clamped to n_train to handle small training sets.
    """
    k_eff = min(k, len(X_train))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", algorithm="auto")
    nn.fit(X_train.astype(np.float64))
    distances, _ = nn.kneighbors(X_test.astype(np.float64))
    return distances[:, -1].astype(np.float64)


def mahalanobis_score(
    X_test: np.ndarray,
    X_train: np.ndarray,
    reg: float = 1e-4,
) -> np.ndarray:
    """
    Mahalanobis distance from the training distribution centroid.
    Ridge regularisation (reg) prevents singularity when n_train < pca_dim.
    Falls back to Euclidean distance if covariance inversion fails.
    """
    X_tr = X_train.astype(np.float64)
    X_te = X_test.astype(np.float64)
    mu   = X_tr.mean(axis=0)

    if X_tr.shape[0] < 2:
        # Not enough samples for covariance — use Euclidean
        return np.linalg.norm(X_te - mu, axis=1)

    cov = np.cov(X_tr.T, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov += np.eye(cov.shape[0]) * reg

    try:
        VI = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        VI = np.linalg.pinv(cov)

    diff   = X_te - mu                                    # (N, d)
    scores = np.sqrt(np.einsum("ni,ij,nj->n", diff, VI, diff))
    return scores.astype(np.float64)


def rank_fuse(*arrays: np.ndarray) -> np.ndarray:
    """
    Rank-normalise each score array to [0,1] then average.
    Produces a combined novelty score that is robust to scale differences
    between GP variance, k-NN distance, and Mahalanobis distance.
    Requires all arrays to have the same length.
    """
    from scipy.stats import rankdata
    if len(arrays) == 1:
        return arrays[0].astype(np.float64)
    n = len(arrays[0])
    normalized = [rankdata(a.astype(np.float64)) / n for a in arrays]
    return np.mean(normalized, axis=0)

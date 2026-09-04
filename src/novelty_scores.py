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


def prototype_distance_score(
    X_test: np.ndarray,
    X_train: np.ndarray,
    type_labels: np.ndarray,
    reg: float = 1e-4,
) -> np.ndarray:
    """
    Per-class Mahalanobis: minimum distance from each test point to the nearest
    per-type distribution (centroid + covariance), not the global centroid.

    Global Mahalanobis (single centroid) misses cases where a novel type lands
    near the global mean but far from all per-type clusters.  This score is
    sensitive to that structure.

    Falls back to Euclidean prototype distance if per-class covariance is
    ill-conditioned (< 2 samples per type).
    """
    X_tr = X_train.astype(np.float64)
    X_te = X_test.astype(np.float64)
    unique_types = np.unique(type_labels)
    min_dists = np.full(len(X_te), np.inf)

    for k in unique_types:
        mask = type_labels == k
        X_k  = X_tr[mask]
        mu_k = X_k.mean(axis=0)

        if X_k.shape[0] >= 2:
            cov_k = np.cov(X_k.T, ddof=1)
            if cov_k.ndim == 0:
                cov_k = np.array([[float(cov_k)]])
            cov_k += np.eye(cov_k.shape[0]) * reg
            try:
                VI_k = np.linalg.inv(cov_k)
                diff = X_te - mu_k
                d    = np.sqrt(np.einsum("ni,ij,nj->n", diff, VI_k, diff))
            except np.linalg.LinAlgError:
                d = np.linalg.norm(X_te - mu_k, axis=1)
        else:
            d = np.linalg.norm(X_te - mu_k, axis=1)

        min_dists = np.minimum(min_dists, d)

    return min_dists.astype(np.float64)


def lof_score(
    X_test: np.ndarray,
    X_train: np.ndarray,
    n_neighbors: int = 5,
) -> np.ndarray:
    """
    Isolation Forest novelty score — better than LOF in high-dimensional
    small-N settings (our regime: 64D, 20–120 training images).

    LOF relies on k-NN local density which degrades in high dimensions
    (distance concentration).  Isolation Forest builds random trees that
    isolate points by recursive splits; anomalies are isolated by fewer
    splits → shorter average path length → higher anomaly score.
    Works well with n_train << n_features and produces more stable scores.

    score_samples returns negative anomaly score (higher = more normal),
    so we negate to get novelty score (higher = more novel).
    """
    from sklearn.ensemble import IsolationForest
    n_est = min(200, max(50, len(X_train) * 2))
    clf = IsolationForest(n_estimators=n_est, random_state=42, contamination="auto")
    clf.fit(X_train.astype(np.float64))
    return (-clf.score_samples(X_test.astype(np.float64))).astype(np.float64)


def zscore_fuse(*arrays: np.ndarray) -> np.ndarray:
    """
    Z-score normalise each score array (jointly across the concatenated
    input — caller must pass the full joint array) then sum.

    Unlike rank_fuse which discards magnitude, zscore_fuse preserves how
    extreme each outlier score is: an image with z=+4 in GPclf space
    contributes more than one with z=+0.5, whereas both would be assigned
    rank/(n-1) ≈ 1.0 by rank_fuse.  This typically improves AUROC when
    signal magnitudes are informative (outliers are not just ranked #1 but
    are genuinely far from the distribution).
    """
    result = np.zeros(len(arrays[0]), dtype=np.float64)
    for a in arrays:
        a = a.astype(np.float64)
        result += (a - a.mean()) / (a.std() + 1e-8)
    return result


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

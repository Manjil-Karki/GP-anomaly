"""Per-fold PCA + standardisation (always fitted on training defectives only)."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class FoldPCA:
    pca: PCA
    scaler: StandardScaler
    n_components: int
    explained_variance_ratio: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the same scaler + PCA fitted on training data."""
        return self.pca.transform(self.scaler.transform(X)).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "FoldPCA":
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_fold_pca(X_train: np.ndarray, n_components: int) -> FoldPCA:
    """
    Fit StandardScaler then PCA on training defective embeddings.
    X_train: (N, D) float32  — training defectives only.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_scaled)
    return FoldPCA(
        pca=pca,
        scaler=scaler,
        n_components=n_components,
        explained_variance_ratio=pca.explained_variance_ratio_.copy(),
    )

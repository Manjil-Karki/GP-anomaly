"""DINOv2 ViT-B/14 feature extraction with on-disk caching.

Embedding strategy: mean of spatial patch tokens, L2-normalised.

Why not the CLS token (original approach)?
  The CLS token summarises global image semantics ("this is a capsule").
  With IMAGE_SIZE=518 and patch_size=14 the model produces 37×37=1369 patch
  tokens — one per 14×14-pixel region.  A defect occupies a small fraction of
  those regions; its signal is heavily diluted in the CLS token but preserved
  in the spatial mean.  Switching to patch-mean embeddings makes the GP's
  input space more sensitive to local texture anomalies.

Why L2-normalise?
  ViT patch tokens lie approximately on a hypersphere.  The GP kernels (RBF,
  Matérn) assume Euclidean distances; applying those to unnormalised tokens
  distorts the geometry.  L2-normalising maps all embeddings to the unit sphere
  so Euclidean distance equals √2·(1 − cos θ) — a monotone function of cosine
  distance — and the kernel length-scale has a consistent interpretation across
  folds and categories.

Cache note: if you have an existing cache built with CLS tokens, delete
  data/embeddings/ before running phase 2 with this version.

torch/torchvision are imported lazily so phases 0-1 work without the GPU stack.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .config import EMBEDDINGS_DIR, IMAGE_SIZE, EMBED_BATCH_SIZE

log = logging.getLogger(__name__)

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


def _get_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_dinov2(device: Optional[str] = None):
    """Load DINOv2 ViT-B/14 backbone from torch.hub (frozen, eval mode)."""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True, verbose=False
    )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info(f"DINOv2 ViT-B/14 loaded on {device}.")
    return model, device


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _path_key(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()


def _patch_mean_embedding(model, x) -> "torch.Tensor":
    """
    Extract L2-normalised mean patch token embedding from a DINOv2 batch.
    x: (B, 3, H, W) tensor already on the correct device.
    Returns (B, 768) float32 tensor on the same device.
    """
    import torch.nn.functional as F
    features = model.forward_features(x)          # dict from DINOv2
    patches  = features["x_norm_patchtokens"]     # (B, n_patches, 768)
    emb      = patches.mean(dim=1)                # (B, 768) — spatial mean
    return F.normalize(emb, p=2, dim=-1)          # unit-sphere L2 norm


def extract_embeddings(
    paths: list[str],
    model,
    device: str,
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    """Return (N, 768) float32 patch-mean L2-normalised embeddings."""
    import torch
    transform = _get_transform()
    all_embs: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        tensors = [transform(Image.open(p).convert("RGB")) for p in batch]
        x = torch.stack(tensors).to(device)
        with torch.no_grad():
            emb = _patch_mean_embedding(model, x)
        all_embs.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(all_embs, axis=0)


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def cache_all_embeddings(
    all_paths: list[str],
    model,
    device: str,
    cache_dir: Path = EMBEDDINGS_DIR,
    batch_size: int = EMBED_BATCH_SIZE,
) -> None:
    """
    Extract and cache embeddings for every image path.
    Each embedding stored as <md5>.npy; index.json maps path → md5.
    Incremental: already-cached paths are skipped.
    """
    import torch
    transform = _get_transform()
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.json"

    index: dict[str, str] = {}
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)

    to_embed = [p for p in all_paths if p not in index]
    log.info(f"Embedding {len(to_embed)} new images ({len(index)} already cached).")

    for i in range(0, len(to_embed), batch_size):
        batch_paths = to_embed[i : i + batch_size]
        tensors = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
        x = torch.stack(tensors).to(device)
        with torch.no_grad():
            embs = _patch_mean_embedding(model, x).cpu().numpy().astype(np.float32)

        for path, emb in zip(batch_paths, embs):
            key = _path_key(path)
            np.save(cache_dir / f"{key}.npy", emb)
            index[path] = key

        if (i // batch_size) % 20 == 0:
            log.info(f"  {i + len(batch_paths)}/{len(to_embed)} embedded")

    with open(index_path, "w") as f:
        json.dump(index, f)
    log.info("Embedding cache updated.")


def load_embeddings(
    paths: list[str],
    cache_dir: Path = EMBEDDINGS_DIR,
) -> np.ndarray:
    """Load (N, 768) float32 array from cache. Raises KeyError if any path missing."""
    index_path = cache_dir / "index.json"
    with open(index_path) as f:
        index: dict[str, str] = json.load(f)

    embs = []
    for p in paths:
        if p not in index:
            raise KeyError(
                f"No cached embedding for:\n  {p}\n"
                "Run phase 2 (cache_all_embeddings) first."
            )
        embs.append(np.load(cache_dir / f"{index[p]}.npy"))
    return np.stack(embs, axis=0)

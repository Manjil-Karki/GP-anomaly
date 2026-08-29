"""DINOv2 ViT-B/14 feature extraction with on-disk caching.

Embedding strategy: CLS token + mean of spatial patch tokens, concatenated
and individually L2-normalised → 1536-dim (2 × 768).

Why concatenate CLS and patch-mean?
  The CLS token encodes global structural context ("this is a screw") while
  the spatial mean of 37×37=1369 patch tokens (one per 14×14-pixel region)
  captures local texture and appearance anomalies.  For LODTO novelty
  detection both signals matter:
    • Global context lets the GP generalise across defect categories that
      share a texture class (e.g. leather, wood).
    • Local patch-mean detects subtle surface defects that are diluted in
      the CLS aggregation.
  Prior ablations showed CLS-only → 0.547 AUROC, patch-mean-only → 0.559.
  Concatenation is expected to recover categories that regressed when
  switching to patch-mean alone (screw, transistor, leather).

Why L2-normalise each part independently before concatenating?
  Both sub-vectors are brought to the unit sphere so neither dominates the
  Euclidean distances used by the GP kernels.  After concatenation the
  combined vector lives on a 1535-dimensional product-of-sphere manifold
  where each half contributes equally to kernel evaluations.

Cache note: delete data/embeddings/ before running phase 2 with this version.

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


def _concat_cls_patch_embedding(model, x) -> "torch.Tensor":
    """
    Extract CLS + patch-mean concatenated embedding from a DINOv2 batch.
    x: (B, 3, H, W) tensor already on the correct device.
    Returns (B, 1536) float32 tensor: [L2(cls) | L2(patch_mean)].
    """
    import torch
    import torch.nn.functional as F
    features   = model.forward_features(x)          # dict from DINOv2
    cls_token  = features["x_norm_clstoken"]        # (B, 768)
    patches    = features["x_norm_patchtokens"]     # (B, n_patches, 768)
    patch_mean = patches.mean(dim=1)                # (B, 768) — spatial mean
    cls_norm   = F.normalize(cls_token,  p=2, dim=-1)
    patch_norm = F.normalize(patch_mean, p=2, dim=-1)
    return torch.cat([cls_norm, patch_norm], dim=-1)  # (B, 1536)


def extract_embeddings(
    paths: list[str],
    model,
    device: str,
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    """Return (N, 1536) float32 CLS+patch-mean L2-normalised embeddings."""
    import torch
    transform = _get_transform()
    all_embs: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        tensors = [transform(Image.open(p).convert("RGB")) for p in batch]
        x = torch.stack(tensors).to(device)
        with torch.no_grad():
            emb = _concat_cls_patch_embedding(model, x)
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
            embs = _concat_cls_patch_embedding(model, x).cpu().numpy().astype(np.float32)

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
    """Load (N, 1536) float32 array from cache. Raises KeyError if any path missing."""
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

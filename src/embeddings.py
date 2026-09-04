"""DINOv2 ViT-L/14 feature extraction with on-disk caching.

Embedding strategy: CLS token + mean of spatial patch tokens, concatenated
and individually L2-normalised → 2048-dim (2 × 1024).

Why ViT-L/14 over ViT-B/14?
  ViT-L/14 has 24 transformer layers vs 12 in ViT-B, 16 attention heads vs 12,
  and 1024-dim tokens vs 768.  The larger model captures finer-grained texture
  and structural differences — critical for the hazelnut and screw categories
  where visually similar defect types overlap in ViT-B/14 embedding space.
  ViT-B/14 produced 0.684 AUROC (fused2); ViT-L/14 expected to push further
  by separating the 12 hard folds that currently have negative entropy gaps.

Why concatenate CLS and patch-mean?
  CLS encodes global context; patch-mean captures local texture anomalies.
  Both are needed for LODTO novelty detection across the 14 MVTec AD categories.

Cache note: embeddings stored in data/embeddings_vitl14/ (separate from ViT-B cache).
Delete this directory to force re-extraction.

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
    """Load DINOv2 ViT-L/14-reg backbone from torch.hub (frozen, eval mode).
    Falls back to dinov2_vitl14 if the registers variant is not in the cached hub repo.
    """
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True, verbose=False
        )
        log.info(f"DINOv2 ViT-L/14-reg loaded on {device}.")
    except Exception as e:
        log.warning(f"dinov2_vitl14_reg unavailable ({e}); falling back to dinov2_vitl14")
        model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitl14", pretrained=True, verbose=False,
            force_reload=True,
        )
        log.info(f"DINOv2 ViT-L/14 (no registers) loaded on {device}.")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
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
    Returns (B, 2048) float32 tensor: [L2(cls) | L2(patch_mean)] for ViT-L/14.
    """
    import torch
    import torch.nn.functional as F
    features   = model.forward_features(x)          # dict from DINOv2
    cls_token  = features["x_norm_clstoken"]        # (B, 1024) for ViT-L
    patches    = features["x_norm_patchtokens"]     # (B, n_patches, 1024)
    patch_mean = patches.mean(dim=1)                # (B, 1024) — spatial mean
    cls_norm   = F.normalize(cls_token,  p=2, dim=-1)
    patch_norm = F.normalize(patch_mean, p=2, dim=-1)
    return torch.cat([cls_norm, patch_norm], dim=-1)  # (B, 2048)


def extract_embeddings(
    paths: list[str],
    model,
    device: str,
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    """Return (N, 2048) float32 CLS+patch-mean L2-normalised embeddings (ViT-L/14)."""
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
    """Load (N, 2048) float32 array from ViT-L/14 cache. Raises KeyError if any path missing."""
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

"""Dataset utilities: manifest construction and image loading."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from .config import DATA_ROOT, MANIFEST_PATH, IMAGE_SIZE


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_manifest(data_root: Path = DATA_ROOT) -> pd.DataFrame:
    """
    Walk MVTec AD directory tree and build a row-per-image DataFrame.
    Columns: category, defect_type, split, image_path, mask_path,
             n_defect_px, n_total_px, severity, width, height.
    """
    records = []
    for cat_dir in sorted(data_root.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        for split in ("train", "test"):
            split_dir = cat_dir / split
            if not split_dir.exists():
                continue
            for defect_dir in sorted(split_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                defect_type = defect_dir.name
                mask_base = cat_dir / "ground_truth" / defect_type
                for img_path in sorted(defect_dir.glob("*.png")):
                    mask_path = mask_base / (img_path.stem + "_mask.png")
                    if not mask_path.exists():
                        mask_path = None
                    n_def, n_tot, sev = _severity(mask_path)
                    try:
                        w, h = Image.open(img_path).size
                    except Exception:
                        w, h = 0, 0
                    records.append({
                        "category":    category,
                        "defect_type": defect_type,
                        "split":       split,
                        "image_path":  str(img_path),
                        "mask_path":   str(mask_path) if mask_path else "",
                        "n_defect_px": n_def,
                        "n_total_px":  n_tot,
                        "severity":    sev,
                        "width":       w,
                        "height":      h,
                    })

    df = pd.DataFrame(records)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_PATH, index=False)
    return df


def load_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["image_path"] = df["image_path"].astype(str)
    df["mask_path"]  = df["mask_path"].fillna("").astype(str)
    return df


def _severity(mask_path: Optional[Path]) -> tuple[int, int, float]:
    if mask_path is None or not Path(str(mask_path)).exists():
        return 0, 0, 0.0
    arr = np.array(Image.open(mask_path))
    n_tot = int(arr.size)
    n_def = int((arr > 127).sum())
    return n_def, n_tot, n_def / n_tot if n_tot else 0.0


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_array(path: str, size: int = IMAGE_SIZE) -> np.ndarray:
    """Load and resize an image to (size, size, 3) uint8 numpy array."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.array(img)

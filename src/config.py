"""Central configuration — paths, hyperparameters, constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Paths ---
DATA_ROOT       = ROOT / "data" / "mvtec"
MANIFEST_PATH   = ROOT / "data" / "manifest.csv"
FOLDS_PATH      = ROOT / "data" / "folds.json"
EMBEDDINGS_DIR  = ROOT / "data" / "embeddings_vitl14"

RESULTS_DIR         = ROOT / "results"
FIGURES_DIR         = RESULTS_DIR / "figures"
GP_RESULTS_DIR      = RESULTS_DIR / "gp_results"
BASELINE_RESULTS_DIR = RESULTS_DIR / "baseline_results"
EVAL_DIR            = RESULTS_DIR / "evaluation"

# --- Image preprocessing ---
IMAGE_SIZE = 518  # DINOv2 ViT-L/14 native resolution (same patch size as ViT-B/14)

# --- PCA sweep ---
PCA_DIMS = [8, 12, 16, 32, 64]

# --- GP optimisation ---
GP_N_RESTARTS = 10
GP_LR         = 0.1
GP_MAX_ITER   = 500

# --- Cost-optimal threshold ---
C_FN = 5.0   # cost of missing a novel defect type
C_FP = 1.0   # cost of a false alarm on a known / normal image

# --- Baselines ---
ENSEMBLE_SIZE      = 5    # deep ensemble MLP members
MC_DROPOUT_PASSES  = 50   # stochastic forward passes
MC_DROPOUT_RATE    = 0.3

# --- MLP autoencoder training ---
MLP_BATCH_SIZE = 128
MLP_EPOCHS     = 50
MLP_HIDDEN     = [256, 128, 64]

# --- Batch sizes ---
EMBED_BATCH_SIZE = 16  # reduced from 32 — ViT-L/14 is ~2.5× larger than ViT-B/14

# --- Evaluation ---
SIGNIFICANCE_LEVEL = 0.05
RANDOM_SEED        = 42

# --- MVTec AD categories ---
# 15 total; toothbrush excluded from LODTO (only 1 defect type → no leave-one-out)
ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]
LODTO_CATEGORIES = [c for c in ALL_CATEGORIES if c != "toothbrush"]

# --- DINOv2 ViT-L/14 output dimension per token ---
# Embedding strategy: [L2(cls) | L2(patch_mean)] → 2 × 1024 = 2048 stored per image
DINO_EMBED_DIM = 2048
